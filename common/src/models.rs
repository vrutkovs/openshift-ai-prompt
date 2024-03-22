use enum_iterator::{all, Sequence};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq)]
pub struct Adapter {
    pub name: &'static str,
    pub path: &'static str,
    pub weight: f64,
}

pub trait Join {
    fn names(&self) -> String;
    fn paths(&self) -> String;
    fn weights(&self) -> String;
}

impl Join for Vec<Adapter> {
    fn names(&self) -> String {
        self.iter().map(|a| a.name).collect::<Vec<&str>>().join(",")
    }

    fn paths(&self) -> String {
        self.iter().map(|a| a.path).collect::<Vec<&str>>().join(",")
    }

    fn weights(&self) -> String {
        self.iter()
            .map(|a| a.weight.to_string())
            .collect::<Vec<String>>()
            .join(",")
    }
}

lazy_static::lazy_static! {
    pub static ref MODELS: Vec<Model> = all::<Model>().collect::<Vec<_>>();
}

#[derive(Serialize, Deserialize, PartialEq, Copy, Clone, Sequence)]
#[serde(tag = "type", content = "details")]
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
    pub fn name(&self) -> &'static str {
        match self {
            Model::StableDiffusionXL => "Stable Diffusion XL",
            Model::Tintin => "Tintin",
            Model::Simpsons => "Simpsons",
            Model::Pixel => "Pixel Art",
            Model::Stickers => "Stickers",
            Model::Impressionism => "Impressionism",
            Model::Lego => "LEGO",
            Model::Ikea => "IKEA",
        }
    }

    pub fn subpath(&self) -> &'static str {
        match self {
            Model::StableDiffusionXL => "xlbase",
            Model::Tintin => "tintin",
            Model::Simpsons => "simpsons",
            Model::Pixel => "pixel-art",
            Model::Stickers => "stickers",
            Model::Impressionism => "impressionism",
            Model::Lego => "lego",
            Model::Ikea => "ikea",
        }
    }

    pub fn base_model(&self) -> Option<Model> {
        match self {
            Model::StableDiffusionXL | Model::Tintin | Model::Simpsons => None,
            _ => Some(Model::StableDiffusionXL),
        }
    }

    pub fn triggers(&self) -> Option<&'static str> {
        match self {
            Model::StableDiffusionXL | Model::Ikea => None,
            Model::Tintin => Some("((herge_style))"),
            Model::Simpsons => Some("((as a simpsons character))"),
            Model::Pixel => Some("((pixel))"),
            Model::Stickers => Some("((sticker))"),
            Model::Impressionism => Some("((in sks style))"),
            Model::Lego => Some("((lego minifig))"),
        }
    }

    pub fn adapters(&self) -> Vec<Adapter> {
        match self {
            Model::StableDiffusionXL | Model::Tintin | Model::Simpsons => vec![],
            Model::Lego => vec![Adapter {
                name: "lego",
                path: "lego/legominifig-v1.0-000003.safetensors",
                weight: 0.8,
            }],
            Model::Ikea => vec![Adapter {
                name: "ikea",
                path: "ikea/ikea_instructions_xl_v1_5.safetensors",
                weight: 1.2,
            }],
            Model::Stickers => vec![Adapter {
                name: "stickers",
                path: "stickers/StickersRedmond.safetensors",
                weight: 0.8,
            }],
            Model::Pixel => vec![Adapter {
                name: "pixel",
                path: "pixel-art/pixel-art-xl.safetensors",
                weight: 1.2,
            }],
            Model::Impressionism => vec![Adapter {
                name: "impressionism",
                path: "impressionism/pytorch_lora_weights.safetensors",
                weight: 1.2,
            }],
        }
    }
}
