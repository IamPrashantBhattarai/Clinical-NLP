"""Pydantic request/response schemas for the dashboard API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw discharge note text")
    age: Optional[int] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="M / F")
    insurance: Optional[str] = Field(None, description="Medicare/Medicaid/Other")
    los_days: Optional[float] = Field(None, description="Length of stay in days")


class PredictResponse(BaseModel):
    probability: float
    predicted_label: int
    risk_level: str
    model_name: str
    feature_type: str
    threshold: float


class ExplainRequest(BaseModel):
    text: str
    top_n: int = 10


class ShapFeature(BaseModel):
    feature: str
    value: float
    shap: float
    direction: str


class ExplainResponse(BaseModel):
    probability: float
    base_value: float
    top_features: List[ShapFeature]
    model_name: str


class ModelResult(BaseModel):
    model: str
    feature_type: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float


class ResultsResponse(BaseModel):
    results: List[ModelResult]
    best: Optional[Dict] = None
    n_models: int
    n_feature_sets: int


class FairnessGroupRow(BaseModel):
    group: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    selection_rate: float


class FairnessAttribute(BaseModel):
    attribute: str
    groups: List[FairnessGroupRow]
    demographic_parity_difference: float
    equalized_odds_difference: float
    fpr_difference: float
    fnr_difference: float


class FairnessResponse(BaseModel):
    attributes: List[FairnessAttribute]


class TopicWord(BaseModel):
    word: str
    weight: float


class Topic(BaseModel):
    topic_id: int
    label: str
    words: List[TopicWord]
    readmission_rate: Optional[float] = None


class TopicsResponse(BaseModel):
    topics: List[Topic]
    coherence_score: Optional[float] = None
    n_topics: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    available_results: bool
