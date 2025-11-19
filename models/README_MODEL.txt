Model files are large binaries. The YOLOv8 face model `yolov8m-face.pt` from the external project is located at:

`g:\Python\Attendance by facial recognition\Cong-Nghe-Xu-Ly-Anh\yolov8m-face.pt`

Please copy it into this folder if you want the main project to use it:

Windows PowerShell example:

```powershell
Copy-Item "g:\Python\Attendance by facial recognition\Cong-Nghe-Xu-Ly-Anh\yolov8m-face.pt" -Destination "g:\Python\Attendance by facial recognition\models\yolov8m-face.pt"
```

After copying, update any code that references the model path to point to `models/yolov8m-face.pt`.