# SDVN10-Anime Cog model

This is an implementation of the [SDVN10-Anime](https://civitai.com/models/254012/sdvn10-anime) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

You can use this template to deploy **any** [CivitAI](https://civitai.com) Stable Diffusion model to [Replicate](https://replicate.com/). Here's how:

## Changing the Model

To use a different model from CivitAI:

1. **Find your desired model on CivitAI.**
2. **Locate the "Download" button for the `safetensors` file of the model.** It's important to use the `safetensors` version for compatibility and security.
3. **Right-click the "Download" button and select "Copy Link Address".** This will copy the direct download URL of the model file.
   ![CivitAI URL](image.png)
4. **Open the `predict.py` file.**
5. **Find the line that defines the `MODEL_LINK` variable.** It will look like this:
   ```python
   MODEL_LINK = "https://civitai.com/api/download/models/286354?type=Model&format=SafeTensor&size=full&fp=fp16"
   ```
6. **Replace the existing URL within the quotes with the link you copied from CivitAI.** Make sure to keep the quotes. For example:
   ```python
   MODEL_LINK = "YOUR_COPIED_LINK_ADDRESS_HERE"
   ```

## Setting Up and Running the Model

**Install the Cog CLI:**

Follow the instructions on the [Cog documentation](https://cog.run) for installing the CLI. A common method is:

```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```

### Install the requirements:

Navigate to the repository directory in your terminal and install the necessary Python packages:

```bash
pip install -r requirements.txt
```

### Download the Pre-trained Weights:

This script will download the model specified by the `MODEL_LINK` in `predict.py` and the necessary `VAE`.

```bash
cog run script/download-weights
```

### Run Predictions (Optional):

Once the weights are downloaded, you can run predictions using the cog predict command. Here's an example:

```bash
cog predict -i prompt="child boy, short hair, crew neck sweater, (masterpiece, best quality:1.6), ghibli, Sun in the sky, Rocky Mountain National Park, Charismatic"
```

## Example Output

![An Example Output](output.png)

## Deploying to Replicate

- Create a new model on [replicate](https://replicate.com/models). Choose a name and visibility for your model.
- Push the model to Replicate using the cog push command. Replicate will provide you with the specific cog push command after you create the model on their platform. It will look something like:

```bash
cog push r8.im/YOUR_USERNAME/YOUR_MODEL_NAME
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
