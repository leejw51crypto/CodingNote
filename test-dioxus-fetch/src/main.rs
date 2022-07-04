#![allow(non_snake_case)]
use dioxus_core::prelude::*;
use dioxus_core_macro::*;
use std::collections::BTreeMap;
use dioxus::prelude::*;
use futures_channel::mpsc::{ UnboundedReceiver};
use futures::StreamExt;
use anyhow::Result;


pub struct MyItem {
    messageid: i64,
    messagetime: String,
    messagetext: String,
}
static MESSAGES: AtomRef<BTreeMap<i64, MyItem>> = |_| BTreeMap::new();
#[derive(Clone, Debug, Default)]
pub struct Program {
    // sqlite database
    // pub db: AtomRef<SqliteDatabase>,
}
impl Program {
    fn new() -> Self {
        Self::default()
    }
    fn initialize(&self)->Result<()> {
        Ok(())
    }
}
pub struct AppProps {
    pub program: Program,
}
#[derive( Props)]
pub struct MainProps {
    taskid: String,
    task: CoroutineHandle<MyCommand>
}

impl PartialEq for MainProps {
    fn eq(&self, other: &Self) -> bool {
        self.taskid == other.taskid
    }
}
#[derive(Clone, Debug, Default)]
pub struct MyCommand {
    command: String,
    message: String,
}

fn start_process(cx: &Scope<AppProps>) -> anyhow::Result<CoroutineHandle<MyCommand>> {
    let names = use_atom_ref(cx, MESSAGES).clone();
    let task = use_coroutine(cx, |mut rx: UnboundedReceiver<MyCommand>| async move {
    

        while let Some(cmd) = rx.next().await {
            match cmd.command.as_str() {
                "fetch" => {

                    for i in 0..10 {
                        let currenttime= chrono::Utc::now();
                        
                        let e = MyItem {
                            messageid: currenttime.timestamp_nanos(),
                            messagetime: "1".into(),
                            messagetext: "2".into()
                        };
                        names.write().insert(e.messageid, e);
                    }
                }
                _ => {
                    println!("unknown command");
                }                          

            } 

        } 
    });

    Ok(task.clone())

}



fn main() {
    let program = Program::new();
    dioxus::desktop::launch_with_props(app, AppProps { program }, |c| c);
}

fn app(cx: Scope<AppProps>) -> Element {

    let task=start_process(&cx).unwrap();
    cx.render(rsx! {
        div { "hello" }
        HomePage{ taskid:"messagestask".into(), task: task}
    })
}

fn HomePage(cx: Scope<MainProps>) -> Element {
    let names = use_atom_ref(&cx, MESSAGES);
    cx.render(rsx! {
        div {
            button {
                onclick: move |_| {
                    cx.props.task.send(MyCommand {
                        command: "fetch".into(),
                        message:"".into(),
                    });  
                },
                "fetch"
            }

            ul {
                names.read().iter().map(|(_key, value)| rsx!{
                    li { "{value.messagetime} {value.messagetext}" }
                })
            }
        }
    })
}
