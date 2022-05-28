use dioxus::prelude::*;
use anyhow::Result;


fn testdb() -> Result<()> {
    let tree = sled::open("my.db")?;
    if let Ok(v2)= tree.get(&"key") {
        if let Some(v) = v2 {
            println!("get data {:?}", v);
        }
    }
    
    let old_value = tree.insert("key", "value")?;
    
    Ok(())
}
fn main() {

    testdb().unwrap();
    dioxus::desktop::launch(app);
}

fn app(cx: Scope) -> Element {
    //let names = ["jim", "bob", "jane", "doe"];
    // make array of 10000 
    let names = (0..10000).map(|a| format!("jim {}",a)).collect::<Vec<_>>();
    let mut count = use_state(&cx, || 0);
    cx.render(rsx! (
        div {
            img { class: "block w-8 h-8",
                src: "https://dev.w3.org/SVG/tools/svgweb/samples/svg-files/ubuntu.svg",
            }

            
            h1 { "count={count}"}
            button {onclick: move |_| count+=1, "+"}
            button {onclick: move |_| count-=1, "-"}


            div { class: "w-full md:w-1/4 p-4 text-center",
                div { class: "w-full relative md:w-3/4 text-center mt-8",
                    button { class: "flex rounded-full",
                        id: "user-menu",
                        "aria_label": "User menu",
                        aria_haspopup: "true",
                        img { class: "h-40 w-40 rounded-full",
                            alt: "",
                            src: "https://lh3.googleusercontent.com/a-/AAuE7mADjk6Ww-zxpX1VxI6Q7f55LSUi1nYUWul2Gdxt=k-s256",
                        }
                    }
                }
            }
            
        
            button {
                class: "check",
                onclick: move |_| {
                    println!("clicked");
                },
                "scan"
            }

           

            img { class: "h-9",
                width: "auto",
                alt: "",
                src: "https://shuffle.dev/yofte-assets/logos/yofte-logo.svg",
            }

            img { class: "block w-8 h-8",
                src: "https://i.imgur.com/ffgW9JQ.png",
            }


            img { class: "block w-8 h-8",
                src: "http://littlesvr.ca/apng/images/clock.gif",
            }

            img { class: "block w-8 h-8",
                src: "http://littlesvr.ca/apng/images/o_sample.png",
            }

            
            
            
            
            
            


            svg { class: "w-6 h-6",
            view_box: "0 0 24 23",
            xmlns: "http://www.w3.org/2000/svg",
            height: "200",
            fill: "none",
            width: "200",
            path { 
                stroke: "black",
                fill: "black",
                d: "M2.01328 18.9877C2.05682 16.7902 2.71436 12.9275 6.3326 9.87096L6.33277 9.87116L6.33979 9.86454L6.3398 9.86452C6.34682 9.85809 8.64847 7.74859 13.4997 7.74859C13.6702 7.74859 13.8443 7.75111 14.0206 7.757L14.0213 7.75702L14.453 7.76978L14.6331 7.77511V7.59486V3.49068L21.5728 10.5736L14.6331 17.6562V13.6558V13.5186L14.4998 13.4859L14.1812 13.4077C14.1807 13.4075 14.1801 13.4074 14.1792 13.4072M2.01328 18.9877L14.1792 13.4072M2.01328 18.9877C7.16281 11.8391 14.012 13.3662 14.1792 13.4072M2.01328 18.9877L14.1792 13.4072M23.125 10.6961L23.245 10.5736L23.125 10.4512L13.7449 0.877527L13.4449 0.571334V1V6.5473C8.22585 6.54663 5.70981 8.81683 5.54923 8.96832C-0.317573 13.927 0.931279 20.8573 0.946581 20.938L0.946636 20.9383L1.15618 22.0329L1.24364 22.4898L1.47901 22.0885L2.041 21.1305L2.04103 21.1305C4.18034 17.4815 6.71668 15.7763 8.8873 15.0074C10.9246 14.2858 12.6517 14.385 13.4449 14.4935V20.1473V20.576L13.7449 20.2698L23.125 10.6961Z",
                stroke_width: "0.35",
            }
            }

            ul {
                names.iter().map(|name| rsx!(
                    div {
                         "{name}" 
                         img { class: "block w-8 h-8",
                             src: "https://i.imgur.com/ffgW9JQ.png",
                         }
                        } 
                ))
            }
        }

    ))
}