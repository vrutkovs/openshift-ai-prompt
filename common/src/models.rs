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
}

impl Model {
    pub fn get_subpath(self) -> String {
        match self {
            Model::StableDiffusionXL => String::from("xlbase"),
            Model::Tintin => String::from("tintin"),
            Model::Simpsons => String::from("simpsons"),
        }
    }

    pub fn get_name(self) -> String {
        match self {
            Model::StableDiffusionXL => String::from("Stable Diffusion XL"),
            Model::Tintin => String::from("Tintin"),
            Model::Simpsons => String::from("Simpsons"),
        }
    }

    pub fn get_basemodel(self) -> Option<Model> {
        match self {
            Model::StableDiffusionXL | Model::Tintin | Model::Simpsons => None,
        }
    }

    pub fn from_subpath(subpath: &str) -> Result<Model, Error> {
        match subpath {
            "xlbase" => Ok(Model::StableDiffusionXL),
            "tintin" => Ok(Model::Tintin),
            "simpsons" => Ok(Model::Simpsons),
            &_ => bail!("Invalid model specified"),
        }
    }
}
