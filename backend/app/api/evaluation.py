from backend.app.evaluation.evaluator import RAGEvaluator
from backend.app.evaluation.schemas import EvaluationReport, EvaluationSample
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluationReport)
async def evaluate_rag(samples: list[EvaluationSample]) -> EvaluationReport:
    try:
        evaluator = RAGEvaluator()
        return evaluator.evaluate(samples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e