"""Ejercicio 2: evaluacion del RAG con Ragas.

    python evaluate_rag.py                 # dataset de ejemplo del enunciado
    python evaluate_rag.py --dataset rag   # ejecuta el RAG real del ejercicio 1

El juez es el mismo LLM configurado en el .env (por defecto el modelo local de
Ollama), asi que la evaluacion no depende de ninguna API de pago.

Nota sobre la API: el enunciado importa `answer_relevance`, que no existe en
Ragas. La metrica se llama `answer_relevancy` (clase `ResponseRelevancy`), y en
0.4.x el dataset se construye con `EvaluationDataset` / `SingleTurnSample`, con
los campos user_input / retrieved_contexts / response / reference.
"""

from __future__ import annotations

import argparse
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

import config

# Dataset de ejemplo tal cual viene en el enunciado.
SAMPLE_DATA = [
    {
        "question": "What is the company policy on remote work?",
        "contexts": [
            "Employees are allowed to work remotely up to 3 days a week. Core hours are 10 AM to 4 PM."
        ],
        "answer": "You can work from home up to three days per week, as long as you are available during core hours.",
        "ground_truth": "Employees can work remotely up to 3 days per week with core hours between 10 AM and 4 PM.",
    },
    {
        "question": "When was the company founded and by whom?",
        "contexts": [
            "The company was incorporated in October 2018.",
            "Our founder, Alice Johnson, started the firm in her garage.",
        ],
        "answer": "The company was founded in October 2018.",
        "ground_truth": "The company was founded in October 2018 by Alice Johnson.",
    },
]


def build_judges():
    """El juez y el modelo de embeddings salen de la misma configuracion que el
    resto del proyecto, para no tener dos proveedores distintos en juego."""
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config.LLM_MODEL,
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            # Los modelos locales no usan el tokenizador de OpenAI, asi que
            # desactivamos el recorte por longitud que hace langchain.
            check_embedding_ctx_length=False,
        )
    )
    return llm, embeddings


def samples_from_sample_data() -> list[SingleTurnSample]:
    return [
        SingleTurnSample(
            user_input=row["question"],
            retrieved_contexts=row["contexts"],
            response=row["answer"],
            reference=row["ground_truth"],
        )
        for row in SAMPLE_DATA
    ]


def samples_from_rag() -> list[SingleTurnSample]:
    """Ejecuta el pipeline del ejercicio 1 sobre preguntas con respuesta conocida.

    Esto es lo que hace util a Ragas: los contextos y las respuestas no son
    inventados, salen de Qdrant y del LLM reales.
    """
    from retrieval import build_messages, retrieve

    questions = json.loads((config.BASE_DIR / "data" / "eval_questions.json").read_text(encoding="utf-8"))
    client = config.get_client()
    samples = []
    for i, item in enumerate(questions, start=1):
        print(f"  ({i}/{len(questions)}) {item['question'][:70]}...")
        chunks = retrieve(item["question"])
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=build_messages(item["question"], chunks),
            temperature=0,
        )
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=[c["raw_text"] for c in chunks],
                response=response.choices[0].message.content,
                reference=item["ground_truth"],
            )
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sample", "rag"), default="sample")
    args = parser.parse_args()

    print(f"[1/3] Construyendo dataset '{args.dataset}'...")
    samples = samples_from_sample_data() if args.dataset == "sample" else samples_from_rag()
    dataset = EvaluationDataset(samples=samples)

    print(f"[2/3] Evaluando {len(samples)} preguntas con juez '{config.LLM_MODEL}'...")
    llm, embeddings = build_judges()
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),                        # groundedness: la respuesta se apoya en el contexto
            ResponseRelevancy(),                   # 'answer_relevancy': responde a lo que se pregunta
            LLMContextPrecisionWithReference(),    # senal/ruido y orden de los chunks
            LLMContextRecall(),                    # el contexto trae todo lo necesario
        ],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    score_columns = [c for c in df.columns if c not in ("user_input", "retrieved_contexts", "response", "reference")]

    print("\n--- Puntuaciones por pregunta ---")
    print(df[["user_input", *score_columns]].to_string(index=False, max_colwidth=45))
    print("\n--- Medias ---")
    print(result)

    # El modo de recuperacion va en el nombre para poder comparar denso vs hibrido.
    sufijo = args.dataset if args.dataset == "sample" else f"rag_{config.RETRIEVAL_MODE}"
    out = config.BASE_DIR / f"resultados_ragas_{sufijo}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n[3/3] Detalle completo guardado en {out}")


if __name__ == "__main__":
    main()
