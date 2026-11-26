"""Front-end conversacional (Chainlit) del RAG.

    chainlit run app.py -w

Flujo por mensaje: vectorizar la pregunta -> recuperar los k chunks mas
cercanos en Qdrant -> montar el prompt con ese contexto -> transmitir la
respuesta del LLM token a token.
"""

import chainlit as cl

from config import LLM_MODEL, TOP_K, get_async_client
from retrieval import build_messages, retrieve


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "**RAG sobre el Big Book of Data Engineering** (paginas 44-58, "
            "capitulo de Lakeflow Connect para SQL Server).\n\n"
            "Pregunta por ejemplo: *¿Que permisos de Unity Catalog necesita "
            "Lakeflow Connect?* — esa respuesta esta en una tabla del PDF."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    # 1. Recuperacion. Se muestra como paso plegable para poder ensenar en la
    #    demo que el contexto viene del PDF y no del conocimiento del modelo.
    async with cl.Step(name=f"Recuperando {TOP_K} fragmentos de Qdrant") as step:
        chunks = await cl.make_async(retrieve)(message.content, k=TOP_K)
        step.output = "\n".join(
            f"[{i}] p.{c['page_number']} (score {c['score']:.3f}) — {c['heading'][:60]}"
            for i, c in enumerate(chunks, start=1)
        ) or "sin resultados"

    if not chunks:
        await cl.Message(content="No hay nada indexado. Ejecuta la ingesta primero.").send()
        return

    # 2. Generacion en streaming.
    answer = cl.Message(content="")
    await answer.send()

    stream = await get_async_client().chat.completions.create(
        model=LLM_MODEL,
        messages=build_messages(message.content, chunks),
        stream=True,
        temperature=0,
    )
    async for part in stream:
        token = part.choices[0].delta.content
        if token:
            await answer.stream_token(token)

    # 3. Fuentes citables, en el panel lateral.
    answer.elements = [
        cl.Text(
            name=f"[{i}] {c['source_file']} p.{c['page_number']}",
            content=c["raw_text"],
            display="side",
        )
        for i, c in enumerate(chunks, start=1)
    ]
    await answer.update()
