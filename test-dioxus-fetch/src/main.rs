#![allow(non_snake_case)]
use dioxus_core::prelude::*;
use dioxus_core_macro::*;
use std::collections::BTreeMap;
use dioxus::prelude::*;
use futures_channel::mpsc::{ UnboundedReceiver};
use futures::StreamExt;
use anyhow::Result;
use rusqlite::{params, Connection};
use std::sync::{Arc,Mutex};
// derive with serde
#[derive(Debug, Clone,Default, serde::Serialize, serde::Deserialize)]
pub struct MyItem {
    messageid: i64,
    messagetime: String,
    messagetext: String,
}
static MESSAGES: AtomRef<BTreeMap<i64, MyItem>> = |_| BTreeMap::new();
#[derive( Debug, Clone)]
pub struct Program {
    db: Arc<Mutex<Connection>>

}
impl Program {
    fn new() -> Self {
        println!("create program");
        let db = Connection::open("db.sqlite").unwrap();
        db.execute("CREATE TABLE IF NOT EXISTS messages (
            messageid INTEGER PRIMARY KEY,
            messagetime TEXT,
            messagetext TEXT
        )", params![]).unwrap();
        Self {
            db:Arc::new(Mutex::new(db))
        }
    }
    async fn get_message(&self)->rusqlite::Result<Vec<MyItem>> {
        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare("SELECT * FROM messages").unwrap();
        let mut rows = stmt.query(params![]).unwrap();
        let mut messages = Vec::new();
        while let Some(row) = rows.next().unwrap() {
            let messageid: i64 = row.get(0)?;
            let messagetime: String = row.get(1)?;
            let messagetext: String = row.get(2)?;
            messages.push(MyItem {
                messageid,
                messagetime,
                messagetext
            });
        }
        Ok(messages)
    }
    fn initialize(&self)->Result<()> {
        let db = self.db.lock().unwrap();
        println!("initialize ----------------------------");
        let count=2;
        for i in 0..count {
            let utctime=chrono::Utc::now();
            let localtime=utctime.with_timezone(&chrono::Local);
            
            let message = MyItem {
                messageid: utctime.timestamp_nanos(),
                messagetime: localtime.to_rfc3339(),
                messagetext: format!("sample text {}", i*10),
            };
            db.execute("INSERT INTO messages (messageid, messagetime, messagetext) VALUES (?1, ?2, ?3)",
                params![message.messageid, message.messagetime, message.messagetext]).unwrap();
        }
        Ok(())
    }
}
pub struct AppProps {
    program: Program,
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
}

/// this is called everytime
fn setup_process(cx: & Scope<AppProps>) -> anyhow::Result<CoroutineHandle<MyCommand>> {
    let names = use_atom_ref(cx, MESSAGES).clone();
    let myprogram= cx.props.program.clone();
    let task = use_coroutine(cx, |mut rx: UnboundedReceiver<MyCommand>| async move {
        while let Some(cmd) = rx.next().await {
            match cmd.command.as_str() {
                "fetch" => {
                    let items=myprogram.get_message().await.unwrap();
                    for e in &items {
                        names.write().insert(e.messageid, e.clone());
                    }
                }
                _ => {
                    println!("unknown command");
                }                          
            } 
        } 
    }); // end of while
    Ok(task.clone())
}

fn main() {
    // setup logic objects, share with props
    // for thread safety, arc + mutex needed
    let program= Program::new();
    program.initialize().unwrap();
    dioxus::desktop::launch_with_props(app, AppProps {program  }, |c| c);
}

// this is called very often
fn app(cx: Scope<AppProps>) -> Element {
    println!("app element----------------------------");
    let task=setup_process(&cx).unwrap();
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
