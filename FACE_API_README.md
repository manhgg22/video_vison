# Face API + Web Camera Check

## Folder database

Tao moi nguoi bang mot folder trong `face_db`:

```text
face_db/
  Manh/
    1.jpg
    2.jpg
  Son Tung/
    1.jpg
```

Anh dang ky nen co 1 khuon mat ro, khong qua mo, khong che mat. Moi nguoi nen co 3-5 anh.

## Cai thu vien

```powershell
pip install -e python-package
pip install opencv-python onnxruntime fastapi uvicorn python-multipart requests
```

Neu dung GPU NVIDIA:

```powershell
pip install onnxruntime-gpu
```

## Chay API

```powershell
python face_api.py --face-db face_db --threshold 0.45
```

API:

```text
GET  /
GET  /health
GET  /people
POST /reload-db
POST /identify
POST /verify/{person_name}
```

Moi lan them anh vao `face_db`, goi:

```powershell
curl -X POST http://127.0.0.1:8000/reload-db
```

## Check bang web

Sau khi chay API, mo trinh duyet:

```text
http://127.0.0.1:8000
```

Trang web se:

- mo camera tren browser
- gui frame len Python API `/identify`
- ve khung mat va hien ten/Unknown ngay tren man hinh
- giu nguyen logic InsightFace + embedding + cosine similarity o backend

## Check bang camera desktop

Mo terminal thu hai:

```powershell
python camera_check.py --api http://127.0.0.1:8000 --camera 0
```

Nhan `q` de thoat camera.

Neu bi nhan nham, tang threshold len `0.50` hoac `0.55`. Neu dung nguoi nhung hay bi Unknown, ha threshold xuong `0.40`.
