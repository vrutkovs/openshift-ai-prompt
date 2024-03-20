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
        let percent = ctx.props().percent;
        let error = ctx.props().error;
        html! {
            <>
                if !message.is_empty() {
                    if error {
                        <Progress description={message.to_string()} value={percent} location={ProgressMeasureLocation::Inside} variant={ProgressVariant::Danger} />
                    }
                }
                if percent > 0.0 && percent != 1.0 {
                    <Progress description={message.to_string()} value={percent} location={ProgressMeasureLocation::Inside}/>
                }
            </>
        }
    }
}
