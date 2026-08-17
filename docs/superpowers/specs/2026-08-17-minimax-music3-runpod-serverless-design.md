# MiniMax-Music3 на RunPod Serverless — спецификация

Дата: 2026-08-17
Статус: утверждено к реализации

## 1. Цель

Развернуть [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) как
serverless-эндпоинт RunPod: JSON-запрос с текстом песни и описанием стиля на входе, готовый
аудиофайл на выходе, оплата только за время генерации, ноль постоянно работающих машин.

Результат работы — репозиторий с Docker-образом воркера, кодом хендлера, тестами, скриптами
подготовки инфраструктуры и инструкцией по деплою.

### Вне области

- Веб-UI и биллинг конечных пользователей.
- Промпт-энхансер (`music-caption-rewriter` из репозитория MiniMax) — отдельная задача.
- Streaming-выдача аудио: внешний API модели принципиально не поддерживает стриминг
  (`stream` должен быть `false`).
- Fine-tuning и любые изменения весов.

## 2. Проверенные факты, на которых стоит дизайн

Всё в этом разделе проверено по первоисточникам 2026-08-17, а не по памяти. Числа отсюда
используются дальше как ограничения.

### 2.1 Модель

| Параметр | Значение | Источник |
|---|---|---|
| Архитектура | Qwen3-backbone 8B (36 слоёв, hidden 4096, GQA 32/8) → depth-decoder 4 слоя (7 RVQ-кодбуков по 1024) → flow-matching DIT 36 слоёв dim 2048 → DAC-декодер | [cookbook](https://raw.githubusercontent.com/sgl-project/sglang-omni/v0.1.2/docs/cookbook/minimax_music3.md) |
| Частота кадров | 25 кадров/с | cookbook |
| Лимит длины | `max_new_tokens` ≤ 9000 кадров = **360 с** | cookbook, HF README |
| Лимит промпта | 5000 токенов после токенизации; контекст 10240 | cookbook |
| Выход sgl-omni | **32 кГц** стерео WAV (сервер ресемплит) | cookbook |
| Выход diffusers | 44.1 кГц стерео (нативный вокодер) | [diffusers docs](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/minimax_music3.md) |
| CFG | Обязателен в обеих стадиях: AR scale 1.5 + top-k 50, акустика 1.7, Euler 30 шагов. Флага отключения нет | cookbook |
| Окно акустики | 200 кадров, хоп 100 | cookbook |
| Детерминизм | Одинаковые lyrics + caption + seed + длина → байт-идентичное аудио. `seed` по умолчанию `0` — фиксированный, не случайный | cookbook |

Важная особенность контракта промпта: **нормализация оставляет от строки, начинающейся с тега,
только сам тег.** `"[Verse] Walking down the street"` превращается в `[start] [verse]`, а текст
теряется молча, без предупреждения. Значит валидация на нашей стороне обязательна.

### 2.2 Рантайм

- Пакет `sglang-omni==0.1.2`, опубликован 2026-08-16, тег `v0.1.2` (commit `0ae7669d6e92`).
  Содержит `sglang_omni/models/minimax_music3/` — поддержка модели есть именно в этом релизе.
- Зависимости 0.1.2: Python <3.13, `torch==2.11.0`, `transformers==5.12.1`, `sglang==0.5.16`,
  `flashinfer_python[cu13]==0.6.14`, `flash-attn-4>=4.0.0b18`, `nixl-cu13`,
  `mooncake-transfer-engine-cuda13`. Стек собран под **CUDA 13**.
- Одна GPU поддерживается (`CUDA_VISIBLE_DEVICES=0`), две — рекомендуемая раскладка
  (AR на устройстве 0, DIT/DAV на устройстве 1). «Требуется две GPU» из HF README — рекомендация
  по пропускной способности, а не жёсткое требование.
- По умолчанию включены: CUDA-граф декода backbone, CUDA-граф RVQ-глубины,
  `torch.compile` для DIT-блоков и DAV-декодера, батчевый seeded-сэмплинг.
- `mem_fraction_static` = 0.50, бюджетирует только KV-кеш backbone; веса акустической стадии
  живут вне этой доли. Понижать не имеет смысла: на 0.35 замерено ~6% медленнее.
- `--max-running-requests` по умолчанию 16, что означает **32 строки декода**, потому что CFG даёт
  каждому запросу вторую строку с собственным KV на всю длину песни. Бюджетировать надо по строкам.
- `--cuda-graph-max-bs` передавать нельзя — модель считает предел сама, переданное значение
  отбрасывается.
- Отклоняемые (не игнорируемые!) параметры запроса: `temperature`, `top_p`, `top_k`,
  `repetition_penalty`, `voice`, `ref_audio`, `ref_text`, `language`, `task_type`, `speed`,
  `stream: true`.

### 2.3 Раскладка весов

`resolve_checkpoint()` в
[`checkpoint.py`](https://raw.githubusercontent.com/sgl-project/sglang-omni/v0.1.2/sglang_omni/models/minimax_music3/checkpoint.py)
ищет от корня модели:

```
<root>/qwen_7B/qwen_7B/                     ~18.5 ГБ (48 шардов + index)
<root>/qwen_7B/qwen3-8B-tokenizer-music/    ~16 МБ
<root>/flowmatching_vae.pth                  9.8 ГБ
<root>/dav.pth                               0.5 ГБ
```

Итого нужно **≈28.8 ГБ** из 57.35 ГБ репозитория. Diffusers-подкаталоги (`transformer/`,
`language_model/`, `condition_encoder/`, `vocoder/`, `rvq_depth_decoder/`) этому рантайму не нужны.
Если указанной раскладки в корне нет, функция дополнительно просматривает подкаталоги на один
уровень вниз — поэтому путь до снапшота HF-кеша подходит как есть.

### 2.4 Платформа

| Ограничение | Значение |
|---|---|
| Payload `/run` | 10 МБ (вход и ответ) |
| Payload `/runsync` | 20 МБ |
| Хранение результата | `/run` — 30 минут, `/runsync` — 1 минута |
| Execution timeout | по умолчанию 600 с, диапазон 5 с … 7 суток |
| Job TTL | по умолчанию 24 ч |
| Idle timeout | по умолчанию 5 с |
| Network volume | монтируется в `/runpod-volume`, привязывает воркеры к одному дата-центру |
| Cached Models | путь `/runpod-volume/huggingface-cache/hub/models--{org}--{name}/snapshots/{hash}/`, время скачивания не биллится, один закешированный репозиторий на эндпоинт, тянется репозиторий целиком |
| S3 API network volume | `https://s3api-{DC}.runpod.io/`, бакет = id volume, прямая загрузка файла ≤500 МБ, дальше multipart; presigned URL не поддерживаются |

Прямое следствие из лимитов: 360-секундный WAV 32 кГц стерео 16 бит — это **46 МБ**, в ответ он не
проходит ни через `/run`, ни через `/runsync`. Значит выдача через бакет — основной путь, а base64 —
только для коротких клипов.

### 2.5 Лицензия — обязательства, а не сноска

`MiniMax-Music3 COMMUNITY LICENSE`:

1. В UI коммерческого продукта или сервиса, использующего модель, обязательна **видимая надпись
   «MiniMax-Music3»**.
2. При совокупной годовой выручке продукта выше **20 млн USD** нужно отдельное предварительное
   письменное разрешение MiniMax (api@minimax.io).
3. Если мы предоставляем третьим лицам сервис, позволяющий генерировать аудио этой моделью, мы
   обязаны внедрить, поддерживать и периодически проверять технические и организационные
   safeguards против нарушающих права выдач, и не ослаблять их.
4. Наследованные лицензии: Qwen3-8B — Apache 2.0, DiT — MIT (stable-audio-tools), VAE — MIT (DAC).

Пункты 1 и 3 — это требования к продукту, который будет вызывать наш эндпоинт. Фиксируем их в
README, чтобы они не потерялись при интеграции.

## 3. Архитектура

### 3.1 Выбранный вариант: sidecar-сервер плюс тонкий async-хендлер

Один контейнер. При инициализации воркера поднимается `sgl-omni serve` дочерним процессом на
`127.0.0.1:8000`. RunPod-хендлер валидирует вход, транслирует его в запрос
`POST /v1/audio/speech`, получает WAV, при необходимости перекодирует, доставляет результат.

```
RunPod job
   │
   ▼
handler.py ──validate──► request_schema.py ──normalize──► lyrics.py
   │
   ├──► server.py (готовность subprocess)
   │
   ▼
POST 127.0.0.1:8000/v1/audio/speech  ──►  sgl-omni  ──►  GPU0: AR   GPU1: DIT+DAV
   │                                                          weights: MODEL_PATH
   ▼ WAV 32 кГц stereo
audio.py (транскод, метаданные)
   │
   ▼
delivery.py ── бакет задан? ──да──► S3/R2 ──► URL
                  └────────────нет──► base64 (с проверкой лимита)
   │
   ▼
{audio_url | audio_base64, duration_s, frames, seed, metrics}
```

Почему так: мы используем публичный, документированный контракт рантайма и получаем даром
continuous batching, CUDA-графы и скомпилированные DIT/DAV. Обновление рантайма — смена одного
пина версии. Цена — HTTP-хоп на localhost (миллисекунды на фоне минут генерации) и необходимость
аккуратно реализовать readiness и корректное завершение процесса.

### 3.2 Отклонённые варианты

**In-process движок** (`engine_builder` импортом). Убирает subprocess и HTTP-хоп, но это внутренний
API без гарантий совместимости; пришлось бы самим собирать запросы и мы потеряли бы планировщик с
батчингом. Выигрыш измеряется миллисекундами, риск — переписыванием на каждом апдейте.

**Веса, запечённые в образ.** Никакой привязки к региону и внешних зависимостей, но образ ~60 ГБ и
медленный первый pull на каждой новой машине. Остаётся задокументированным fallback'ом (§9.2).

## 4. Компоненты

Каждый модуль — одна ответственность и тестируется отдельно.

### `src/config.py`
Читает окружение в неизменяемый датакласс `Settings`, валидирует на старте. Единственное место,
где живут значения по умолчанию. Полный список переменных — §8.

### `src/model_path.py`
`resolve_model_path(settings) -> str`. Порядок:

1. `MODEL_PATH`, если задан и существует — использовать как есть.
2. Снапшот Cached Models: глоб `{HF_HOME}/hub/models--MiniMaxAI--MiniMax-Music3/snapshots/*/`;
   при нескольких снапшотах берём последний по mtime.
3. `MODEL_REPO_ID` как строка (sgl-omni скачает сам) — только для локальной отладки на Pod.

На каждом кандидате проверяем присутствие четырёх артефактов из §2.3 и внятно логируем, какой
путь выбран и почему остальные отброшены. Отсутствие весов — ошибка старта, а не первого запроса.

### `src/server.py`
Управление жизненным циклом дочернего процесса.

- `start()`: собирает argv (`sgl-omni serve --model-path … --host 127.0.0.1 --port …` плюс
  `SGL_EXTRA_ARGS`), выставляет `CUDA_VISIBLE_DEVICES` из `GPU_COUNT` (`"0"` или `"0,1"`),
  запускает процесс, стримит его stdout/stderr в наш логгер с префиксом.
- `wait_ready(timeout)`: поллинг `GET /v1/models` с интервалом 2 с. Одновременно проверяем, что
  процесс жив: если он умер, поднимаем ошибку сразу, а не ждём таймаут.
- `is_alive()`, `stop()`: `SIGTERM`, затем `SIGKILL` по таймауту.
- Запуск синхронный на импорте модуля хендлера — RunPod не должен принимать задачи до готовности.

### `src/request_schema.py`
`parse(job_input) -> GenerationRequest`. Обязательные `lyrics` и `prompt` (непустые). Опциональные
`duration` (1…`MAX_DURATION_S`) либо `max_new_tokens` (1…9000) — взаимоисключающие;
`duration` конвертируется как `round(duration * 25)`. Плюс `seed` (неотрицательное 64-битное,
по умолчанию 0), `format` (`wav|mp3|opus|flac`), `bitrate`.

Явно отклоняем поля из §2.2 с сообщением «параметр не поддерживается этой моделью, темп задаётся
в prompt» — так ошибка клиента видна сразу, а не превращается в тихо другой результат. Принимаем
алиасы `input` → `lyrics` и `instructions` → `prompt` для совместимости с примерами MiniMax.

### `src/lyrics.py`
`normalize(lyrics) -> (text, warnings)`. Находит строки, которые начинаются со структурного тега и
содержат после него ещё текст, и разбивает их на две строки. Возвращает предупреждения, которые
попадают в ответ в `warnings[]`. Это защита от тихой потери текста песни (§2.1).

### `src/audio.py`
`transcode(wav_bytes, format, bitrate) -> (bytes, AudioInfo)` на PyAV (`av` уже в зависимостях
рантайма). `format="wav"` — passthrough с разбором заголовка. `AudioInfo` содержит длительность,
частоту, число каналов, число кадров модели (`round(duration * 25)`).

### `src/delivery.py`
`deliver(data, info, job_id, settings) -> dict`. Если заданы `BUCKET_*` — загрузка через
`runpod.serverless.utils.rp_upload` и возврат `audio_url`. Иначе base64 с проверкой размера
**после** кодирования против `BASE64_MAX_ENCODED_BYTES`; при превышении — ошибка с явной
рекомендацией настроить бакет и указанием фактического и допустимого размера. Обрезать ответ
нельзя ни при каких условиях.

Имена переменных `BUCKET_ENDPOINT_URL` / `BUCKET_ACCESS_KEY_ID` / `BUCKET_SECRET_ACCESS_KEY` — это
конвенция `runpod-python`; при реализации сверить с исходниками `runpod==1.12.0` и, если конвенция
изменилась, использовать актуальную (задача плана реализации).

### `src/handler.py`
Async-хендлер. Стадии с `runpod.serverless.progress_update`: `validating` → `generating` →
`encoding` → `uploading`. HTTP-запрос к sgl-omni через `httpx.AsyncClient` с таймаутом
`GENERATION_TIMEOUT_S`. `concurrency_modifier` возвращает `MAX_CONCURRENCY`. Ловим
`asyncio.CancelledError`, отменяем HTTP-запрос и пробрасываем дальше, чтобы GPU освобождалась при
отмене задачи. Собирает `metrics` по стадиям.

### `src/logging_setup.py`
Структурные JSON-логи в stdout: `job_id`, стадия, длительности, коды ошибок. Без текста песни и
промптов на уровне INFO — они могут быть пользовательскими данными; на DEBUG допустимо.

## 5. Контракт API

### Запрос

```json
{
  "input": {
    "lyrics": "[Verse]\nWalking down the empty street at midnight\n[Chorus]\nAnd I keep on walking",
    "prompt": "A melancholic lo-fi hip-hop track at 85 BPM in F minor: mellow Rhodes piano, soft vinyl crackle, dusty boom-bap drums, warm upright bass. Intimate bedroom production.",
    "duration": 30,
    "seed": 42,
    "format": "mp3",
    "bitrate": "192k"
  }
}
```

| Поле | Тип | По умолчанию | Ограничения |
|---|---|---|---|
| `lyrics` (алиас `input`) | string | — | обязательное, непустое; теги на отдельных строках |
| `prompt` (алиас `instructions`) | string | — | обязательное, непустое |
| `duration` | number | 30 | 1…`MAX_DURATION_S` (360); взаимоисключимо с `max_new_tokens` |
| `max_new_tokens` | int | — | 1…9000 |
| `seed` | int | 0 | ≥0, 64-битное |
| `format` | string | `DEFAULT_FORMAT` (`mp3`) | `wav`, `mp3`, `opus`, `flac` |
| `bitrate` | string | `192k` | только для `mp3`/`opus` |

### Успешный ответ

```json
{
  "audio_url": "https://…/{job_id}.mp3",
  "format": "mp3",
  "sample_rate": 32000,
  "channels": 2,
  "duration_s": 30.0,
  "frames": 750,
  "seed": 42,
  "warnings": [],
  "metrics": {"validate_ms": 2, "generate_ms": 41230, "encode_ms": 380, "upload_ms": 540}
}
```

Вместо `audio_url` может быть `audio_base64` — в зависимости от конфигурации доставки. `duration_s`
может быть меньше запрошенного: модель заканчивает песню сама по audio-end токену, и это не
усечение.

### Ошибки

```json
{"error": "unsupported_parameter: 'temperature' is fixed for this model", "code": "unsupported_parameter"}
```

| Код | Причина | Ретрай |
|---|---|---|
| `invalid_request` | пустые lyrics/prompt, длина вне диапазона, конфликт `duration`/`max_new_tokens` | нет |
| `unsupported_parameter` | параметр из списка отклоняемых (§2.2) | нет |
| `upstream_rejected` | 4xx от sgl-omni, текст пробрасывается | нет |
| `generation_failed` | 5xx от sgl-omni | да, `refresh_worker` |
| `engine_unavailable` | процесс умер или не поднялся | да, `refresh_worker` |
| `result_too_large` | base64 сверх лимита и бакет не настроен | нет |
| `timeout` | генерация дольше `GENERATION_TIMEOUT_S` | нет |

## 6. Обработка ошибок и деградация

- **Процесс не поднялся за `SERVER_STARTUP_TIMEOUT_S`** — падаем на инициализации, до приёма
  задач. Воркер, который принял задачу и не может её выполнить, хуже воркера, который не стартовал.
- **Процесс умер во время работы** — `engine_unavailable` + `refresh_worker: true`; RunPod поднимет
  свежий воркер. Не пытаемся перезапускать subprocess внутри живого контейнера: состояние GPU
  после OOM ненадёжно.
- **4xx от sgl-omni** пробрасываем как есть: это ошибка запроса, ретрай её не исправит.
- **Отмена задачи** — `asyncio.CancelledError` → отмена HTTP-запроса → пробросить. Без этого GPU
  остаётся занятой отменённой генерацией.
- **Результат не влезает в base64** — явная ошибка с числами, никогда не обрезанный аудиофайл.

## 7. Docker-образ

Двухслойная сборка.

**`docker/base.Dockerfile`** — вендоренный upstream `docker/Dockerfile` с тега `v0.1.2`. Он уже
пинит по digest `lmsysorg/sglang@sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7`
и flashinfer-кеш `hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101`,
собирает UCX 1.20 (commit `d8e50df`) и ставит зависимости. Вендорим, а не наследуемся от
`lmsysorg/sglang-omni:dev`, по двум причинам: тег `dev` не пинится и обновлён 2026-06-16, то есть
раньше релиза поддержки Music 3; а его entrypoint при каждом старте контейнера клонирует репозиторий
с GitHub — в serverless это недопустимая сетевая зависимость на пути cold start.

**`docker/Dockerfile`** — наш слой:

```dockerfile
FROM minimax-music3-base:v0.1.2
ENV SGLANG_OMNI_AUTO_CLONE=0
RUN uv pip install --system --no-deps "sglang-omni==0.1.2" && \
    uv pip install --system "runpod==1.12.0" "httpx" && \
    python3 -c "import sglang_omni.models.minimax_music3 as m; print(m.__file__)"
COPY src/ /app/src/
ENV PYTHONUNBUFFERED=1 HF_HOME=/runpod-volume/huggingface-cache
ENTRYPOINT ["python3", "-u", "/app/src/handler.py"]
```

Проверка импорта на этапе сборки обязательна: она ловит расхождение между зафиксированными в базовом
образе зависимостями и требованиями 0.1.2 на сборке, а не на первом cold start в продакшене.

### 7.1 Прогретые кеши

Upstream кладёт в образ FlashInfer JIT-кеш, провалидированный **только на Ada (SM89) и Hopper
(SM90a)**; на других архитектурах он компилируется в свой каталог при первом использовании. Плюс
sgl-omni по умолчанию компилирует DIT и DAV через `torch.compile`. И то и другое на непрогретой
машине происходит внутри оплачиваемого cold start.

Отсюда два решения:

1. GPU фиксируем на SM89/SM90a (§8.1) — используем уже готовый кеш.
2. `scripts/warm_caches.sh` запускается на Pod с целевой картой: поднимает сервер, гоняет короткую
   генерацию, упаковывает `~/.cache/flashinfer` и каталог Inductor (`TORCHINDUCTOR_CACHE_DIR`) в
   архив. `docker/caches.Dockerfile` добавляет архив слоем поверх образа воркера. Шаг
   опциональный, но обязателен для продакшена: без него первый запрос на свежей машине платит за
   компиляцию.

## 8. Конфигурация

### 8.1 Эндпоинт RunPod

| Настройка | Значение | Почему |
|---|---|---|
| GPUs / worker | **2** | AR на устройстве 0, DIT+DAV на устройстве 1 |
| Тип GPU | **L40S / RTX 6000 Ada (48 ГБ, SM89)**; приоритет-альтернатива **H100 (80 ГБ, SM90a)** | прогретый FlashInfer-кеш есть только под SM89/SM90a; A100 (SM80) и A6000/A40 (SM86) дают компиляцию на первом запросе |
| CUDA-фильтр | 13.x | зависимости 0.1.2 собраны под cu13 |
| Execution timeout | 1800 с | дефолтных 600 с не хватит на 360-секундный трек |
| Idle timeout | 120–300 с | cold start дорогой, держим воркер тёплым между запросами |
| Active workers | 0 для dev, ≥1 для латентно-чувствительного прода | иначе каждый первый запрос платит cold start |
| Max workers | по нагрузке, начать с 3 | |
| Scaling | queue delay 4 с | генерация длинная, агрессивный скейлинг не нужен |
| Cached model | `MiniMaxAI/MiniMax-Music3` | время скачивания не биллится |
| Network volume | **не прикреплять** | конфликт точки монтирования `/runpod-volume` с Cached Models |

Cached Models и собственный network volume монтируются в один и тот же `/runpod-volume`, поэтому
это взаимоисключающие варианты. Выбран Cached Models; network volume — fallback из §9.2.

### 8.2 Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `MODEL_PATH` | — | явный путь к весам, обходит автопоиск |
| `MODEL_REPO_ID` | `MiniMaxAI/MiniMax-Music3` | id репозитория для поиска снапшота и fallback-скачивания |
| `HF_HOME` | `/runpod-volume/huggingface-cache` | корень кеша Cached Models |
| `GPU_COUNT` | `2` | раскладка `CUDA_VISIBLE_DEVICES` |
| `SGL_PORT` | `8000` | порт localhost-сервера |
| `SGL_EXTRA_ARGS` | — | доп. флаги, например `--max-running-requests 32` |
| `SERVER_STARTUP_TIMEOUT_S` | `1200` | бюджет на загрузку весов и графы |
| `GENERATION_TIMEOUT_S` | `1500` | таймаут HTTP-запроса, ниже execution timeout эндпоинта |
| `MAX_CONCURRENCY` | `1` | сколько задач воркер берёт параллельно |
| `MAX_DURATION_S` | `360` | верхняя граница `duration` |
| `DEFAULT_FORMAT` | `mp3` | |
| `DEFAULT_BITRATE` | `192k` | |
| `BASE64_MAX_ENCODED_BYTES` | `9500000` | запас под лимит `/run` в 10 МБ |
| `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`, `BUCKET_SECRET_ACCESS_KEY` | — | заданы → выдача URL, не заданы → base64 |
| `LOG_LEVEL` | `INFO` | |

### 8.3 Про `MAX_CONCURRENCY`

По умолчанию 1: предсказуемая латентность, простая отладка, детерминированный расход VRAM.
Батчинг — сильная сторона этой модели (depth-decoder стоит одинаково при одном и восьми запросах в
батче, а AR-шаг ограничен чтением весов), поэтому поднятие до 4–8 заметно улучшает стоимость на
трек при высокой нагрузке. Но CFG делает каждый запрос двумя строками KV, поэтому поднимать
`MAX_CONCURRENCY` нужно вместе с `--max-running-requests` и с замером VRAM: считаем строки, а не
запросы. Порядок изменения задокументируем в README, дефолт не меняем без данных бенчмарка.

## 9. Риски и fallback

| Риск | Вероятность | Реакция |
|---|---|---|
| Cached Models не принимает репозиторий на 57 ГБ или тянет его слишком долго | средняя | fallback §9.2: свой network volume с 28.8 ГБ подмножества |
| Зависимости базового образа расходятся с требованиями `sglang-omni==0.1.2` | средняя | проверка импорта на сборке; при поломке — собрать базовый слой из `pyproject.toml` тега `v0.1.2` напрямую |
| 2× SM89/SM90a недоступны в выбранном ДЦ | средняя | добавить второй тип GPU в приоритеты; при переходе на SM80/86 обязательно прогреть кеши (§7.1) под эту арх. |
| Cold start дольше приемлемого | высокая | прогретые кеши + `idle timeout` 300 с + ≥1 active worker |
| Имена `BUCKET_*` в `runpod==1.12.0` отличаются от конвенции | низкая | сверить с исходниками при реализации |
| Стоимость 2 GPU/воркер выше ожиданий | средняя | сравнить с одногпушной раскладкой на 1× H100 80 ГБ по данным бенчмарка — код это уже поддерживает через `GPU_COUNT=1` |

### 9.2 Fallback на network volume

`scripts/prepare_network_volume.sh` создаёт volume в ДЦ с нужными GPU и заливает через S3 API
только нужное подмножество (§2.3, ≈28.8 ГБ): `qwen_7B/qwen_7B/`,
`qwen_7B/qwen3-8B-tokenizer-music/`, `flowmatching_vae.pth`, `dav.pth`. Файлы больше 500 МБ идут
multipart. Дальше эндпоинт настраивается без Cached Models, с прикреплённым volume, а `MODEL_PATH`
указывается явно. Код воркера при этом не меняется — в этом и смысл `model_path.py`.

## 10. Производительность

Единственный референс из документации: пять треков по 30 с отрендерены на одной H200 с дефолтными
настройками; конкретных цифр времени апстрим не даёт. Поэтому производительность — предмет замера,
а не предположения.

`scripts/benchmark.py` замеряет и печатает таблицу:

- секунды аудио на секунду wall-clock при `max_new_tokens` = 250 / 750 / 1500 / 9000;
- то же при `MAX_CONCURRENCY` = 1 / 4 / 8 с соответствующим `--max-running-requests`;
- пиковую VRAM на каждом устройстве;
- cold start: до готовности сервера и до первого байта первого ответа, отдельно на прогретых и
  непрогретых кешах.

Критерии приёмки заполняются числами после первого прогона и фиксируются в README как базовая
линия для регрессий. Заранее фиксируем только требования, которые не зависят от железа:
30-секундный трек должен укладываться в `GENERATION_TIMEOUT_S`, а 360-секундный — в execution
timeout эндпоинта; если нет — поднимаем таймауты, а не режем длину.

## 11. Тестирование

GPU в CI нет, поэтому тесты разделены по уровням.

**Юнит-тесты (CI, без GPU).** `request_schema`: обязательные поля, границы, конфликт
`duration`/`max_new_tokens`, каждый отклоняемый параметр, алиасы `input`/`instructions`.
`lyrics.normalize`: текст на строке тега разбивается и даёт warning; корректная лирика не меняется;
регистр тегов; теги без текста. `model_path.resolve`: выбор снапшота, несколько снапшотов,
неполный набор артефактов, явный `MODEL_PATH`. `audio.transcode`: короткий синтетический WAV во все
форматы, разбор длительности. `delivery`: ветка бакета с замоканным `rp_upload`, ветка base64,
превышение лимита.

**Интеграционные (CI, без GPU).** Хендлер против фейкового HTTP-сервера, отдающего заранее
записанный WAV: полный путь, коды ошибок, 4xx и 5xx апстрима, отмена задачи, поля `metrics`.
Плюс `python src/handler.py --test_input "$(cat test_input.json)"` — локальный прогон RunPod SDK.

**Сборка (CI).** `docker build` обоих слоёв, проверка импорта модели внутри образа.

**На реальной GPU (руками, до деплоя).** На Pod с 2× L40S: поднять сервер, прогнать
`scripts/smoke_test.py` на 10-секундном клипе, проверить детерминизм (два запроса с одним seed →
идентичные байты), затем 360-секундный трек. После деплоя — `smoke_test.py` против эндпоинта через
`/run` с поллингом.

Порядок работы — TDD: тест на поведение, затем реализация.

## 12. Структура репозитория

```
minimax_music_3/
├─ README.md                      # деплой, конфигурация, лицензионные требования, базовая линия перф.
├─ docker/
│  ├─ base.Dockerfile             # вендоренный upstream v0.1.2 (пины по digest)
│  ├─ Dockerfile                  # слой воркера
│  └─ caches.Dockerfile           # слой с прогретыми FlashInfer/Inductor кешами
├─ src/
│  ├─ handler.py  server.py  request_schema.py  lyrics.py
│  ├─ audio.py    delivery.py  model_path.py    config.py  logging_setup.py
├─ tests/
│  ├─ test_request_schema.py  test_lyrics.py  test_model_path.py
│  ├─ test_audio.py  test_delivery.py  test_handler.py
│  └─ fixtures/short.wav
├─ scripts/
│  ├─ warm_caches.sh              # прогрев кешей на Pod
│  ├─ prepare_network_volume.sh   # fallback-заливка весов через S3 API
│  ├─ smoke_test.py               # проверка Pod или задеплоенного эндпоинта
│  └─ benchmark.py                # замеры перф. и VRAM
├─ test_input.json
├─ requirements.txt               # runpod, httpx (+dev: pytest, respx)
├─ .dockerignore
└─ docs/superpowers/specs/2026-08-17-minimax-music3-runpod-serverless-design.md
```

Язык: спецификация на русском, код, комментарии и README — на английском.

## 13. Порядок работ

1. Скелет проекта, `config.py`, `logging_setup.py`, CI на юнит-тестах.
2. `request_schema.py` + `lyrics.py` с тестами — чистые функции, никакого GPU.
3. `model_path.py` с тестами на временных каталогах.
4. `audio.py`, `delivery.py` с тестами.
5. `server.py` + `handler.py`, интеграционные тесты против фейкового сервера.
6. `docker/base.Dockerfile`, `docker/Dockerfile`, сборка и проверка импорта.
7. Прогон на Pod с 2× L40S: smoke, детерминизм, полная длина.
8. `warm_caches.sh` + `caches.Dockerfile`, повторный замер cold start.
9. Деплой эндпоинта, `smoke_test.py` через API, `benchmark.py`, запись базовой линии в README.
10. `prepare_network_volume.sh` — только если Cached Models не отработает.
