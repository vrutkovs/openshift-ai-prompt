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
}

impl Model {
    pub fn get_subpath(self) -> String {
        match self {
            Model::StableDiffusionXL => String::from("xlbase"),
            Model::Tintin => String::from("tintin"),
            Model::Simpsons => String::from("simpsons"),
            Model::Pixel => String::from("pixel-art"),
        }
    }

    pub fn get_name(self) -> String {
        match self {
            Model::StableDiffusionXL => String::from("Stable Diffusion XL"),
            Model::Tintin => String::from("Tintin"),
            Model::Simpsons => String::from("Simpsons"),
            Model::Pixel => String::from("Pixel Art"),
        }
    }

    pub fn get_basemodel(self) -> Option<Model> {
        match self {
            Model::StableDiffusionXL | Model::Tintin | Model::Simpsons => None,
            Model::Pixel => Some(Model::StableDiffusionXL),
        }
    }

    pub fn from_subpath(subpath: &str) -> Result<Model, Error> {
        match subpath {
            "xlbase" => Ok(Model::StableDiffusionXL),
            "tintin" => Ok(Model::Tintin),
            "simpsons" => Ok(Model::Simpsons),
            "pixel-art" => Ok(Model::Pixel),
            &_ => bail!("Invalid model specified"),
        }
    }

    pub fn append_trigger_words(self) -> Option<String> {
        match self {
            Model::StableDiffusionXL => None,
            Model::Tintin => Some(String::from("((herge_style))")),
            Model::Simpsons => Some(String::from("((as a simpsons character))")),
            Model::Pixel => Some(String::from("((pixel))")),
        }
    }

    pub fn get_additional_adapter_weight(self) -> Vec<f64> {
        match self {
            Model::StableDiffusionXL | Model::Tintin | Model::Simpsons => vec![],
            Model::Pixel => vec![1.2],
        }
    }

    pub fn get_additional_adapter_names(self) -> Vec<String> {
        match self {
            Model::StableDiffusionXL | Model::Tintin | Model::Simpsons => vec![],
            Model::Pixel => vec![String::from("pixel-art")],
        }
    }

    pub fn get_additional_adapter_path(self) -> Vec<String> {
        match self {
            Model::StableDiffusionXL | Model::Tintin | Model::Simpsons => vec![],
            Model::Pixel => vec![String::from("pixel-art/pixel-art-xl.safetensors")],
        }
    }
}
