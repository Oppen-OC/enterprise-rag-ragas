# Marker: requisitos y solución de problemas

Solo hacen falta para extraer con Marker, el backend por defecto de `ingest_pdf.py` (los resultados de `docs/informe.md` salen de él). `--backend pymupdf` no los necesita.

`marker-pdf==2.0.0` arrastra `surya-ocr`, que **no corre los modelos en el
proceso de Python**: levanta un contenedor Docker `vllm/vllm-openai`
(`surya-vllm-<puerto>`) y le habla por HTTP. Por tanto necesita:

- **Docker Desktop** arrancado y la imagen `vllm/vllm-openai` (varios GB, se
  descarga sola la primera vez).
- **GPU NVIDIA** con VRAM libre para ese servidor. `pip` instala por defecto la
  build de PyTorch para CPU; hay que sustituirla:

  ```powershell
  .\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision
  .\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0 torchvision
  .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
  ```

  Sustituye `cu130` por la versión de CUDA que soporte tu driver (`nvidia-smi`).

- Que **no haya un contenedor `surya-vllm-*` viejo** ocupando la GPU. Si lo hay,
  el arranque del nuevo agota su espera de 600 s y falla con
  `SpawnError: vllm server failed to become healthy`. Se limpia con
  `docker rm -f $(docker ps -aq --filter name=surya-vllm)`.

Sin GPU, o si no quieres pelearte con esto, `--backend pymupdf` extrae en
segundos y sin modelos, a cambio de partir las celdas multilínea de las tablas
(ver [informe.md](informe.md), sección 2).

## Rehacer la extracción

```powershell
.\.venv\Scripts\python.exe ingest_pdf.py dataaw\lakeflow-connect-sqlserver.pdf --page-offset 43                     # Marker (requisitos arriba)
.\.venv\Scripts\python.exe ingest_pdf.py dataaw\lakeflow-connect-sqlserver.pdf --page-offset 43 --force --backend pymupdf   # sin GPU ni Docker
```
