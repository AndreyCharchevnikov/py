import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import Response
from pathlib import Path
from tempfile import NamedTemporaryFile
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

app = FastAPI()

@app.post("/pdfs/images/qwen", status_code=status.HTTP_200_OK)
async def ocr_screenshot_endpoint(
    file: UploadFile = File(...),
):
    """
    Принимает скриншот, прогоняет его через модель Qwen2.5-VL и возвращает полученный текст в формате markdown.
    Файл сохраняется во временный файл для обработки, затем удаляется.
    """
    try:
        # Чтение файла и сохранение во временный файл
        file_bytes = await file.read()
        if not file_bytes:
            raise Exception("Пустой файл")
        with NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = Path(tmp_file.name)
    except Exception as e:
        print("Ошибка при сохранении временного файла:", e)
        raise HTTPException(status_code=500, detail="Ошибка при сохранении временного файла")

    try:
        # Открываем изображение
        image = Image.open(tmp_file_path)

        # Формируем сообщение для модели
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        "Извлеки весь текст и таблицы из этой страницы документа и представь их в формате Markdown. "
                        "Не включай изображения."
                    )}
                ]
            }
        ]
        
        # Загружаем модель и процессор
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            attn_implementation="flash_attention_2", 
            low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

        # Подготовка входных данных для инференса
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = [image]  # Передаем изображение как список
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Генерация ответа
        generated_ids = model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    except Exception as e:
        print("Ошибка при выполнении OCR:", e)
        raise HTTPException(status_code=500, detail="Ошибка при выполнении OCR")
    finally:
        # Удаляем временный файл и очищаем память GPU
        tmp_file_path.unlink(missing_ok=True)
        torch.cuda.empty_cache()

    return Response(content=output_text[0], media_type="text/markdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
