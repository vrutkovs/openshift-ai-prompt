use anyhow::{bail, Error};
use enum_iterator::Sequence;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct ModelType {
    subpath: String,
    name: String,
    base_model: Option<Model>,
}

#[derive(Serialize, Deserialize, Copy, Clone, Sequence, PartialEq)]
#[serde(rename_all_fields(serialize = "lowercase", deserialize = "lowercase"))]
pub enum Model {
    StableDiffusionXL,
    Tintin,
    Simpsons,
    Pixel,
    Stickers,
    Impressionism,
    Lego,
    Ikea,
}

impl Model {
    pub fn get_subpath(self) -> String {
        match self {
            Model::StableDiffusionXL => String::from("xlbase"),
            Model::Tintin => String::from("tintin"),
            Model::Simpsons => String::from("simpsons"),
            Model::Pixel => String::from("pixel-art"),
            Model::Stickers => String::from("stickers"),
            Model::Impressionism => String::from("impressionism"),
            Model::Lego => String::from("lego"),
            Model::Ikea => String::from("ikea"),
        }
    }

    pub fn get_name(self) -> String {
        match self {
            Model::StableDiffusionXL => String::from("Stable Diffusion XL"),
            Model::Tintin => String::from("Tintin"),
            Model::Simpsons => String::from("Simpsons"),
            Model::Pixel => String::from("Pixel Art"),
            Model::Stickers => String::from("Stickers"),
            Model::Impressionism => String::from("Impressionism"),
            Model::Lego => String::from("LEGO"),
            Model::Ikea => String::from("IKEA"),
        }
    }

    pub fn get_basemodel(self) -> Option<Model> {
        match self {
            Model::StableDiffusionXL | Model::Tintin | Model::Simpsons => None,
            _ => Some(Model::StableDiffusionXL),
        }
    }

    pub fn from_subpath(subpath: &str) -> Result<Model, Error> {
        match subpath {
            "xlbase" => Ok(Model::StableDiffusionXL),
            "tintin" => Ok(Model::Tintin),
            "simpsons" => Ok(Model::Simpsons),
            "pixel-art" => Ok(Model::Pixel),
            "stickers" => Ok(Model::Stickers),
            "impressionism" => Ok(Model::Impressionism),
            "lego" => Ok(Model::Lego),
            "ikea" => Ok(Model::Ikea),
            &_ => bail!("Invalid model specified"),
        }
    }

    pub fn append_trigger_words(self) -> Option<String> {
        match self {
            Model::StableDiffusionXL | Model::Ikea => None,
            Model::Tintin => Some(String::from("((herge_style))")),
            Model::Simpsons => Some(String::from("((as a simpsons character))")),
            Model::Pixel => Some(String::from("((pixel))")),
            Model::Stickers => Some(String::from("((sticker))")),
            Model::Impressionism => Some(String::from("((in sks style))")),
            Model::Lego => Some(String::from("((lego minifig))")),
        }
    }

    pub fn get_additional_adapter_weight(self) -> Vec<f64> {
        match self {
            Model::StableDiffusionXL | Model::Tintin | Model::Simpsons => vec![],
            Model::Lego => vec![0.8],
            Model::Stickers => vec![0.8, 1.0],
            Model::Pixel => vec![1.2, 1.0],
            Model::Impressionism => vec![1.2, 1.0],
            _ => vec![1.2],
        }
    }

    pub fn get_additional_adapter_names(self) -> Vec<String> {
        match self {
            Model::StableDiffusionXL | Model::Tintin | Model::Simpsons => vec![],
            Model::Lego => vec![String::from("lego")],
            Model::Ikea => vec![String::from("ikea")],
            Model::Stickers => vec![String::from("stickers"), String::from("lora")],
            Model::Pixel => vec![String::from("pixel-art"), String::from("lora")],
            Model::Impressionism => vec![String::from("impressionism"), String::from("lora")],
        }
    }

    pub fn get_additional_adapter_path(self) -> Vec<String> {
        match self {
            Model::StableDiffusionXL | Model::Tintin | Model::Simpsons => vec![],
            Model::Lego => vec![String::from("lego/legominifig-v1.0-000003.safetensors")],
            Model::Ikea => vec![String::from("ikea/ikea_instructions_xl_v1_5.safetensors")],
            Model::Stickers => vec![
                String::from("stickers/StickersRedmond.safetensors"),
                String::from("lora/pytorch_lora_weights.safetensors"),
            ],
            Model::Pixel => vec![
                String::from("pixel-art/pixel-art-xl.safetensors"),
                String::from("lora/pytorch_lora_weights.safetensors"),
            ],
            Model::Impressionism => vec![
                String::from("impressionism/pytorch_lora_weights.safetensors"),
                String::from("lora/pytorch_lora_weights.safetensors"),
            ],
        }
    }
}
