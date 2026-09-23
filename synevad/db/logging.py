import os
import re
import signal
import urllib.error
import urllib.request
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig

# At import, not per call: `load_dotenv` does not override an existing variable, but it
# does repopulate one a caller deliberately unset, so calling it inside the resolver made
# `MLFLOW_TRACKING_URI` impossible to clear for the process's lifetime.
load_dotenv()

_CREDENTIALS = re.compile(r"://[^/@]*@")


def redact_tracking_uri(uri: str) -> str:
    """`http://user:pass@host:5000` with the userinfo removed, for logs and errors.

    Every message about the tracking store carries the URI, and a URI that authenticates
    to the tracking server carries a password with it.
    """
    return _CREDENTIALS.sub("://<credentials>@", uri)


def tracking_uri_from_config(config: DictConfig) -> str:
    """Resolve the MLflow tracking URI for write and read paths.

    Prefer `MLFLOW_TRACKING_URI` (the tracking server, which serializes the concurrent
    writers of a multi-GPU sweep). Fall back to the historical SQLite file under
    `logging.root` for single-process runs.
    """
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    root = Path(config.logging.root).resolve()
    return f"sqlite:///{root / config.logging.db_name}"


def is_tracking_server_uri(uri: str) -> bool:
    """Whether the URI addresses an MLflow tracking server rather than a store directly.

    Only the server takes concurrent writers: it owns the one connection pool the sweep's
    workers queue behind, so no worker talks to the backing database itself.
    """
    return uri.startswith(("http://", "https://"))


CONNECT_TIMEOUT_SECONDS = 5


def check_tracking_store(uri: str) -> None:
    """Raise unless the server both answers on its socket and serves an MLflow read.

    The `/health` probe is what makes this fast. MLflow retries a failed request with
    exponential backoff, so a server that is down takes minutes to report — silently, and
    once per worker if the check is left to the workers. The MLflow read still has to
    follow it: a server that is up can still be pointed at a store it cannot query, and
    that is the failure that would otherwise land after the first model finished training.
    """
    try:
        urllib.request.urlopen(
            f"{uri.rstrip('/')}/health", timeout=CONNECT_TIMEOUT_SECONDS
        )
    except urllib.error.HTTPError:
        # It answered, just not with 200 (an auth-gated server, say). Reachability is all
        # this probe is for; the MLflow read below is what decides whether it is usable.
        pass

    mlflow.set_tracking_uri(uri)
    mlflow.search_experiments(max_results=1)


def artifact_location(config: DictConfig) -> str | None:
    """Where a newly created experiment stores its artifacts; `None` lets the server pick.

    Against a tracking server, `None` resolves to `mlflow-artifacts:/<experiment_id>`:
    clients upload through the server over HTTP, so the files land on the host that serves
    the UI. Naming a `file://` path instead makes every worker write to *its own* disk at
    that path while the server records the location as if it owned it — the artifacts then
    exist only on whichever machine ran the sweep, and read back as missing anywhere else.
    Only the SQLite fallback, where the writing process is the store, names a local dir.
    """
    if is_tracking_server_uri(tracking_uri_from_config(config)):
        return None
    root = Path(config.logging.root).resolve()
    artifact_dir = root / config.logging.artifacts_dirname
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir.as_uri()


def get_or_create_experiment(name: str, artifact_location: str | None) -> str:
    """Experiment id for `name`, creating it only if no concurrent writer got there first.

    Every worker of a multi-GPU sweep runs this at startup, so the plain check-then-create
    is racy: on an experiment name that does not exist yet, two workers both read `None`
    and the loser's create is rejected for the name the winner already took. Under
    `mp.spawn(join=True)` that one exception tears down the entire sweep before the first
    model trains. A single-process sweep could never hit it.
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    try:
        return mlflow.create_experiment(name, artifact_location=artifact_location)
    except MlflowException:
        # Lost the race. The winner's row is committed by the time our INSERT is
        # rejected, so the second lookup is the one that resolves it.
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            raise
        return exp.experiment_id


def end_run_on_termination() -> None:
    """End the active run as KILLED when the process is signalled, then die normally.

    `mp.spawn(join=True)` SIGTERMs the surviving workers the moment one of them raises,
    and Ctrl-C reaches all of them at once. MLflow only closes the active run from an
    `atexit` hook, which neither signal runs, so every worker that was mid-run leaves a
    row stuck in RUNNING in the shared store — rows a later `finished_only` read silently
    drops instead of reporting. Sequential SQLite sweeps never produced them.
    """
    owner_pid = os.getpid()

    def handler(signum, _frame):
        # DataLoader workers are forked from here and inherit this handler; only the
        # process that owns the MLflow run may end it.
        if os.getpid() == owner_pid and mlflow.active_run() is not None:
            mlflow.end_run(status="KILLED")
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)


def setup_mlflow(config: DictConfig):
    mlflow.set_tracking_uri(tracking_uri_from_config(config))

    exp_id = get_or_create_experiment(
        config.logging.experiment_name, artifact_location(config)
    )
    mlflow.set_experiment(experiment_id=exp_id)


def log_params_and_tags(set_name: str, config: DictConfig) -> None:
    tags = get_tags(set_name, config)
    mlflow.set_tags(tags)

    params = get_params(set_name, config)
    mlflow.log_params(params)


def declared_params(config: DictConfig) -> dict[str, str]:
    """The `params.*` a run of `config` carries, exactly as MLflow will store them.

    Split out of `get_params` because `synevad.eval.resume` compares these strings against
    what the store returned, so it has to build them through the same function that wrote
    them. `str(ListConfig)` renders `image_size` as `[256, 256]`; a hand-rolled
    `str(list(...))` or `json.dumps` would drift by a space and silently re-run a sweep
    that is already complete.
    """
    return {key: str(val) for key, val in config.params.items()}


def declared_tags(config: DictConfig) -> dict[str, str]:
    """The `tags.*` block `config` declares, without the per-eval-set runtime tags.

    These are the tags that identify the sweep point rather than the run: they are the
    same for every eval set the config is scored on.
    """
    return {key: str(val) for key, val in config.tags.items()}


def get_params(set_name: str, config: DictConfig) -> dict[str, str]:
    return declared_params(config)


def get_tags(set_name: str, config: DictConfig) -> dict[str, str]:
    tags = declared_tags(config)
    tags["set_name"] = set_name
    if set_name != "real":
        tags["severity"] = config.data.synthetic[set_name].severity
        tags["scorer_result"] = config.data.synthetic[set_name].scorer_result
    return tags
