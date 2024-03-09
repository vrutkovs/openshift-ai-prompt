use yew::prelude::*;

pub enum Msg {
    Update,
}

pub struct ProgressBar {
    message: String,
    value: f64,
}
impl Component for ProgressBar {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        Self {
            message: "".to_string(),
            value: 0.0,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        true
    }

    fn view(&self, _ctx: &Context<Self>) -> Html {
        let value = self.value;
        html! {
            <progress class="progress is-primary" value={value.to_string()} max=1.0>
                { format!("{:.0}%", 100.0 * value) }
            </progress>
        }
    }
}
