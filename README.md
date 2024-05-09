# Virtual Influencer Project

Welcome to our Virtual Influencer project, where artificial intelligence brings a new dimension to social media. This project involves the creation of an AI-driven virtual influencer from Armenia who autonomously travels the world, shares fashion trends, interacts with followers, and manages social media posts and captions.

## Project Overview

We utilize several cutting-edge technologies to bring our virtual influencer to life:

- **Realistic_Vision_V2.0 Model**: We use the Realistic_Vision_V2.0 model from Stable Diffusion to ensure the authenticity of our virtual influencer's appearance.
- **Deepfaking**: Integration of the influencer's face into various video scenarios.
- **Automatic Captioning**: Generating relevant captions based on the image content.
- **DreamBooth**: Maintaining consistent identity across various images through fine-tuning techniques.

## Key Features

- **Image and Caption Generation**: Automatically produce and caption high-quality images.
- **Video Creation**: Utilize deepfake technology for dynamic video content.
- **Minimal Human Oversight**: Fully automated posting and interaction process on Instagram.
- **Identity Consistency**: Use of DreamBooth to ensure the influencer's identity remains consistent across different scenarios.

## Installation

Clone the repository to get started:

```bash
git clone https://github.com/username/virtual_influencer.git
cd virtual_influencer 
```

## Usage


## Usage

### Image and Caption Generation

For generating images, captions, and translating text, follow these steps:

1. **Open the Jupyter Notebook**: Navigate to the Jupyter notebook file that is set up for generating images and captions.
    ```
    jupyter notebook Instagram_Content_Generation_for_Virtual_Influencer.ipynb
    ```

2. **Generate Content**: Once inside the notebook, you can change the prompts as needed to generate various images and captions. This allows for customization based on different themes or requirements.

3. **Save Outputs**: The generated images and captions are automatically saved in the downloads directory under corresponding names, ensuring easy access and organization.

4. **Translate Text**: If you need to translate the generated captions or any other text into another language, follow the instructions provided in the notebook. This might involve setting parameters for the translation model or specifying the target language.

### Video Deepfaking

To create deepfake videos, use the following steps:

1. **Open the Video Deepfaking Notebook**:
    ```
    jupyter notebook Video_DeepFaking.ipynb
    ```

2. **Specify Input and Target**: Within the notebook, specify the input image and the target video for your deepfake scenario.

### Audioio generation and Lip Syncing with Wav2Lip

We have tried to combine audio generation with OpenAi Whisper and lipsyncing with Wav2lip. Although our initial results with Wav2Lip did not turn out as expected, you can try to replicate in the following notebook where further instructions will be found:

1. **Open the Lip Syncing Notebook**:
    ```
    jupyter notebook LipSyncing_Wav2Lip.ipynb
    ```

### Fine-Tuning Model

For image generation, you can use already pretrained weights available in drive: https://drive.google.com/drive/folders/12vZ9cIQ89yR2x8wxRFFdF4g7YB2QLQu5?usp=drive_link 

To fine-tune the model with DreamBooth, ensure you have GPU resources available, such as those provided by Google Colab:

1. **Open the Fine-Tuning Notebook**:

    ```bash
    jupyter notebook dreambooth_fine_tuning_rv.ipynb
    ```

2. **Execute the Training Command**: Run the following command within the notebook to start the training process:

    ```bash
    !python3 train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --seed=1337 \
    --resolution=512 \
    --train_batch_size=1 \
    --train_text_encoder \
    --mixed_precision="fp16" \
    --use_8bit_adam \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=40 \
    --sample_batch_size=4 \
    --max_train_steps=600 \
    --save_interval=200 \
    --save_sample_prompt="photo of zwx girl" \
    --concepts_list="concepts_list.json"
    ```

### Automated Posting on Instagram

Automatically post generated images and captions by following these instructions:

1. **Open the Posting Notebook**:

    ```bash
    jupyter notebook Automated_Image_Posting_on_Instagram.ipynb
    ```

2. **Prepare the Posts**: Ensure that the `caption.txt` file is moved to the same directory as the `.ipynb` file. Keep the generated image in the downloads folder for easy access.

The steps are designed to guide users through using the project’s functionalities, from image and caption generation to posting on social media platforms.




