# AURA Vision API — Frontend & Website Integration Guide

This guide provides everything needed for frontend and backend engineers to integrate the **AURA Diamond Ring ML Metadata Extraction & Visual Search API** into an existing website or application.

---

## ⚡ Quick Start for Developers

- **API Base URL**: `http://localhost:8000` (or your deployed server domain)
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Alternative ReDoc**: `http://localhost:8000/redoc`
- **CORS Status**: Enabled for all origins (`*`) — you can call this API directly from any browser client without CORS proxy issues.

---

## 💎 Primary Endpoint: Extract Image Metadata

### `POST /api/v1/extract-metadata`
Upload an image of a diamond ring to instantly extract detailed visual attributes, zero-shot classification, and bounding box coordinates.

- **URL**: `http://localhost:8000/api/v1/extract-metadata`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

#### Request Parameters (Form-Data)
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File (Binary) | **Yes** | Ring image (`.jpg`, `.jpeg`, `.png`, `.webp`) |
| `crop_x1` | Float | No | Optional left crop coordinate |
| `crop_y1` | Float | No | Optional top crop coordinate |
| `crop_x2` | Float | No | Optional right crop coordinate |
| `crop_y2` | Float | No | Optional bottom crop coordinate |

#### Example JSON Response
```json
{
  "success": true,
  "metadata": {
    "diamond_shape": "Oval",
    "shape": "Oval",
    "band_metal_tone": "14K Yellow Gold",
    "band_color": "14K Yellow Gold",
    "color": "14K Yellow Gold",
    "band_architecture": "Classic Straight Shank",
    "band_configuration": "Classic Straight Shank",
    "band_type": "Plain Solitaire Band",
    "prong_setting": "Classic 4-Prong Setting",
    "prong_style": "Classic 4-Prong Setting",
    "prong_color": "14K Yellow Gold",
    "ring_style": "Solitaire",
    "style": "Solitaire",
    "confidence_scores": {
      "diamond_shape": 0.98,
      "band_metal_tone": 0.96,
      "band_architecture": 0.94,
      "prong_setting": 0.95,
      "ring_style": 0.96
    },
    "all_probabilities": {
      "metal_probabilities": {
        "14K Yellow Gold": 0.96,
        "14K White Gold": 0.03,
        "14K Rose Gold": 0.01
      }
    }
  },
  "region_detection": {
    "confidence": 0.96,
    "detected_region": {
      "top": 42.0,
      "left": 55.0,
      "bottom": 440.0,
      "right": 425.0,
      "width": 370.0,
      "height": 398.0,
      "rel_top": 0.084,
      "rel_left": 0.11,
      "rel_bottom": 0.88,
      "rel_right": 0.85
    },
    "all_regions": []
  },
  "image_dimensions": {
    "width": 500,
    "height": 500
  },
  "inference_time_ms": 138.4
}
```

---

## 🌐 Alternative Endpoint: Extract Metadata via Image URL

### `POST /api/v1/metadata/extract-url`
Extract metadata from a remote image URL or base64 data string.

- **URL**: `http://localhost:8000/api/v1/metadata/extract-url`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Body
```json
{
  "image_url": "https://example.com/images/engagement-ring.jpg"
}
```

---

## 🔍 Visual Catalog Search (Metadata + Matching Products)

### `POST /api/v1/search/image`
Extracts metadata AND returns ranked catalog matches.

- **URL**: `http://localhost:8000/api/v1/search/image`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: (Required, File)
  - `top_k`: (Optional, int, default: `12`)
  - `shape`: (Optional filter, e.g. `"Round"`, `"Oval"`)
  - `band_color`: (Optional filter, e.g. `"14K Yellow Gold"`)
  - `style`: (Optional filter, e.g. `"Solitaire"`, `"Halo"`)

---

## 💻 Frontend Code Examples

### 1. Vanilla JavaScript / HTML (Direct `<input type="file">` Upload)

```html
<!DOCTYPE html>
<html>
<head>
  <title>Ring Metadata Extractor</title>
</head>
<body>
  <input type="file" id="ringFileInput" accept="image/*" />
  <pre id="outputResult"></pre>

  <script>
    const fileInput = document.getElementById("ringFileInput");
    const output = document.getElementById("outputResult");

    fileInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append("file", file);

      try {
        output.innerText = "Extracting metadata...";
        const response = await fetch("http://localhost:8000/api/v1/extract-metadata", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) throw new Error("API request failed");
        const data = await response.json();
        
        console.log("Extracted Ring Metadata:", data.metadata);
        output.innerText = JSON.stringify(data.metadata, null, 2);
      } catch (err) {
        console.error("Error:", err);
        output.innerText = "Failed: " + err.message;
      }
    });
  </script>
</body>
</html>
```

---

### 2. React / Next.js Component Example

```jsx
import React, { useState } from "react";

export default function RingMetadataUploader({ onMetadataExtracted }) {
  const [loading, setLoading] = useState(false);
  const [metadata, setMetadata] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/api/v1/extract-metadata", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (data.success) {
        setMetadata(data.metadata);
        if (onMetadataExtracted) {
          onMetadataExtracted(data.metadata);
        }
      }
    } catch (error) {
      console.error("Failed to extract metadata:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ring-uploader-card">
      <input type="file" accept="image/*" onChange={handleFileChange} />
      
      {preview && <img src={preview} alt="Ring preview" style={{ maxWidth: 200 }} />}
      
      {loading && <p>Analyzing ring attributes...</p>}

      {metadata && (
        <div className="metadata-display">
          <h3>Detected Ring Attributes:</h3>
          <ul>
            <li><strong>Shape:</strong> {metadata.diamond_shape} ({Math.round(metadata.confidence_scores.diamond_shape * 100)}%)</li>
            <li><strong>Metal:</strong> {metadata.band_metal_tone} ({Math.round(metadata.confidence_scores.band_metal_tone * 100)}%)</li>
            <li><strong>Shank:</strong> {metadata.band_architecture}</li>
            <li><strong>Prong:</strong> {metadata.prong_setting}</li>
            <li><strong>Style:</strong> {metadata.ring_style}</li>
          </ul>
        </div>
      )}
    </div>
  );
}
```

---

### 3. Node.js / Backend (Axios + Form-Data)

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function extractRingMetadata(imageFilePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(imageFilePath));

  const response = await axios.post('http://localhost:8000/api/v1/extract-metadata', form, {
    headers: form.getHeaders(),
  });

  return response.data;
}

// Example usage
extractRingMetadata('./sample_ring.jpg')
  .then(res => console.log('Metadata:', res.metadata))
  .catch(err => console.error(err));
```

---

### 4. Python Client (`requests`)

```python
import requests

url = "http://localhost:8000/api/v1/extract-metadata"
files = {"file": open("sample_ring.jpg", "rb")}

response = requests.post(url, files=files)
data = response.json()

print(f"Diamond Shape: {data['metadata']['diamond_shape']}")
print(f"Metal Tone:    {data['metadata']['band_metal_tone']}")
print(f"Ring Style:    {data['metadata']['ring_style']}")
print(f"Shank Type:    {data['metadata']['band_architecture']}")
print(f"Prong Setting: {data['metadata']['prong_setting']}")
```

---

### 5. cURL Command (Terminal Test)

```bash
curl -X POST "http://localhost:8000/api/v1/extract-metadata" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_ring.jpg;type=image/jpeg"
```

---

## 🛠️ How to Run the API Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API Server**:
   ```bash
   python run_server.py
   ```
   Or using Uvicorn directly:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Open the Documentation**:
   Visit `http://localhost:8000/docs` in your browser.
