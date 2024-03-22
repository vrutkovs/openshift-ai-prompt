use patternfly_yew::prelude::*;
use yew::prelude::*;

#[derive(Clone, Properties, PartialEq)]
pub struct Props {
    pub message: AttrValue,
    pub percent: f64,
    pub error: bool,
}

pub struct ProgressBar;
impl Component for ProgressBar {
    type Message = ();
    type Properties = Props;

    fn create(_: &Context<Self>) -> Self {
        Self
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let message = &ctx.props().message;
        let value = ctx.props().percent;
        let error = ctx.props().error;
        let mut variant = ProgressVariant::Default;
        if !message.is_empty() && error {
            variant = ProgressVariant::Danger;
        }
        html! {
            <>
                if value > 0.0 && value != 1.0 {
                    <Progress description={message.to_string()} {value} location={ProgressMeasureLocation::Inside} {variant}/>
                }
            </>
        }
    }
}
