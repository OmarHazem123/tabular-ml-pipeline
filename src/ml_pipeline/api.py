"""FastAPI inference service.

Endpoints: ``GET /health``, ``GET /model-info``, ``POST /predict``, ``POST /predict/batch``.

Model resolution at start-up (first match wins):

1. an explicitly injected model (tests);
2. ``$ML_PIPELINE_MODEL_DIR``: an exported model dir (``model.joblib`` + ``metadata.json``),
   which is what the Docker image uses;
3. the ``production`` model of the registry at ``$ML_PIPELINE_REGISTRY_DIR`` (default
   ``artifacts/registry``).

The artifact's SHA-256 is verified before it is deserialised. If no model can be loaded the
service still starts: ``/health`` reports ``unavailable`` with HTTP 503 and the prediction
endpoints return 503, so orchestrators can tell "process up" from "ready to serve".

(No ``from __future__ import annotations`` here: FastAPI resolves endpoint annotations
at runtime.)
"""

import logging
import math
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ml_pipeline import __version__
from ml_pipeline.config import CategoricalFeature, NumericFeature, PipelineConfig, load_config
from ml_pipeline.exceptions import PipelineError
from ml_pipeline.model import ChurnModel, Prediction
from ml_pipeline.registry import ModelMetadata, ModelRegistry, RegistryError, load_model_dir
from ml_pipeline.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureInfo,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    requests_to_frame,
)

ENV_MODEL_DIR = "ML_PIPELINE_MODEL_DIR"
ENV_REGISTRY_DIR = "ML_PIPELINE_REGISTRY_DIR"
ENV_CONFIG = "ML_PIPELINE_CONFIG"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

logger = logging.getLogger("ml_pipeline.api")


@dataclass
class ServiceState:
    config: PipelineConfig
    model: ChurnModel | None = None
    metadata: ModelMetadata | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class LoadedModel:
    model: ChurnModel
    metadata: ModelMetadata


def _resolve(model_dir: Path | None, registry_dir: Path) -> tuple[ChurnModel, ModelMetadata]:
    if model_dir is not None:
        return load_model_dir(model_dir)
    loaded = ModelRegistry(registry_dir).load_production()
    if loaded is None:
        raise RegistryError(f"no production model in registry {registry_dir.as_posix()}")
    return loaded


def _state(request: Request) -> ServiceState:
    state: ServiceState = request.app.state.service
    return state


def require_model(request: Request) -> LoadedModel:
    state = _state(request)
    if state.model is None or state.metadata is None:
        raise HTTPException(
            status_code=503, detail=f"model unavailable: {state.error or 'not loaded'}"
        )
    return LoadedModel(model=state.model, metadata=state.metadata)


ModelDep = Annotated[LoadedModel, Depends(require_model)]


def json_safe(value: Any) -> Any:
    """Replace NaN/Infinity (which Python's JSON parser accepts) with strings, so a 422
    that echoes the offending input can itself be serialised as strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value


def _result(customer_id: str | None, prediction: Prediction) -> PredictionResult:
    return PredictionResult.model_validate({"customer_id": customer_id, **prediction.model_dump()})


def create_app(
    *,
    config: PipelineConfig | None = None,
    model_dir: Path | None = None,
    registry_dir: Path | None = None,
    model: tuple[ChurnModel, ModelMetadata] | None = None,
) -> FastAPI:
    """Application factory (``uvicorn --factory ml_pipeline.api:create_app``)."""
    config = config or load_config(os.environ.get(ENV_CONFIG) or None)
    if model_dir is None and os.environ.get(ENV_MODEL_DIR):
        model_dir = Path(os.environ[ENV_MODEL_DIR])
    if registry_dir is None:
        registry_dir = Path(os.environ.get(ENV_REGISTRY_DIR) or config.paths.registry_dir)
    serving = config.serving
    resolved_registry = registry_dir

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = ServiceState(config=config)
        app.state.service = state
        try:
            churn_model, metadata = model or _resolve(model_dir, resolved_registry)
            churn_model.warm_up()
            state.model, state.metadata = churn_model, metadata
            logger.info(
                "Loaded model %s (%s, sha256 %s)",
                metadata.model_id,
                metadata.model_name,
                metadata.artifact.sha256[:12],
            )
        except PipelineError as exc:
            state.error = str(exc)
            logger.error("Serving without a model: %s", exc)
        yield

    app = FastAPI(
        title="Telco Churn Inference API",
        version=__version__,
        description=(
            "Calibrated churn probabilities with SHAP reason codes and input-drift warnings."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = json_safe(jsonable_encoder(exc.errors()))
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        logger.info(
            "request_id=%s method=%s path=%s status=%d latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse, "description": "No model loaded"}},
    )
    def health(request: Request) -> JSONResponse:
        state = _state(request)
        loaded = state.model is not None and state.metadata is not None
        body = HealthResponse(
            status="ok" if loaded else "unavailable",
            model_loaded=loaded,
            model_id=state.metadata.model_id if state.metadata else None,
            detail=state.error,
            uptime_seconds=round(time.monotonic() - state.started_at, 3),
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=200 if loaded else 503)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info(loaded: ModelDep) -> ModelInfoResponse:
        metadata, reference = loaded.metadata, loaded.model.reference
        features: list[FeatureInfo] = []
        for spec in config.data_schema.features:
            if isinstance(spec, CategoricalFeature):
                features.append(
                    FeatureInfo(name=spec.name, kind=spec.kind, categories=list(spec.categories))
                )
                continue
            observed = reference.numeric.get(spec.name)
            features.append(
                FeatureInfo(
                    name=spec.name,
                    kind=spec.kind,
                    min_value=spec.min_value if isinstance(spec, NumericFeature) else 0.0,
                    max_value=spec.max_value if isinstance(spec, NumericFeature) else 1.0,
                    training_min=observed.min if observed else None,
                    training_max=observed.max if observed else None,
                )
            )
        dataset = metadata.lineage.dataset
        return ModelInfoResponse(
            model_id=metadata.model_id,
            model_name=metadata.model_name,
            stage=metadata.stage,
            created_at=metadata.created_at,
            threshold=metadata.threshold,
            calibration_method=metadata.calibration_method,
            artifact_sha256=metadata.artifact.sha256,
            git_commit=metadata.lineage.git.commit,
            config_fingerprint=metadata.lineage.config_fingerprint,
            dataset_sha256=dataset.sha256 if dataset else None,
            cv=metadata.cv,
            validation=metadata.validation,
            test=metadata.test,
            promotion_approved=metadata.promotion.approved if metadata.promotion else None,
            features=features,
        )

    TopK = Annotated[
        int,
        Query(ge=1, le=serving.max_top_k, description="Number of SHAP reason codes to return"),
    ]

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest, loaded: ModelDep, top_k: TopK = serving.default_top_k
    ) -> PredictionResponse:
        started = time.perf_counter()
        frame = requests_to_frame([payload], config.data_schema)
        [prediction] = loaded.model.predict_records(frame, top_k)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        logger.info(
            "prediction model_id=%s probability=%.4f churn=%s warnings=%d latency_ms=%.2f",
            loaded.metadata.model_id,
            prediction.churn_probability,
            prediction.churn_prediction,
            len(prediction.warnings),
            latency_ms,
        )
        return PredictionResponse(
            **_result(payload.customer_id, prediction).model_dump(),
            model_id=loaded.metadata.model_id,
            latency_ms=latency_ms,
        )

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(
        payload: BatchPredictionRequest, loaded: ModelDep, top_k: TopK = serving.default_top_k
    ) -> BatchPredictionResponse:
        started = time.perf_counter()
        frame = requests_to_frame(payload.instances, config.data_schema)
        predictions = loaded.model.predict_records(frame, top_k)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        logger.info(
            "batch_prediction model_id=%s n=%d positives=%d latency_ms=%.2f",
            loaded.metadata.model_id,
            len(predictions),
            sum(p.churn_prediction for p in predictions),
            latency_ms,
        )
        return BatchPredictionResponse(
            model_id=loaded.metadata.model_id,
            n_predictions=len(predictions),
            latency_ms=latency_ms,
            predictions=[
                _result(instance.customer_id, prediction)
                for instance, prediction in zip(payload.instances, predictions, strict=True)
            ],
        )

    return app
