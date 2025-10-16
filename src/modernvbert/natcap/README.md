# NatCap 📄
Here you can find the code used to generate the dataset `NatCap`. It requires `google.genai` python SDK. You can install it as

```bash
pip install google-genai
```

Next you will need a Google API Key to make the calls to gemini models.

```bash
export GOOGLE_API_KEY=MY_KEY
```

## Usage

To annotate one HF dataset with columns `(image,text)` you can use:

```bash
python async_annotate.py <dataset_hf_id> --batch_size 8
```

> ⚠️ This uses parallel API calling, tweak the `batch_size` argument depending on your system.


