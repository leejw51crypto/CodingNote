use bevy::input::mouse::MouseButtonInput;
use bevy::{
    prelude::*,
    text::{BreakLineOn, JustifyText},
};
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use bevy_mod_picking::prelude::*;
use rand::Rng;

#[derive(Component, Clone)]
struct Title {
    text: String,
    font_size: f32,
    color: Color,
    position: Vec2,
    max_width: f32,
}

#[derive(Component)]
struct Block {
    number: u64,
    position: Vec2,
    width: f32,
    height: f32,
}

#[derive(Component, Clone)]
struct Transaction {
    width: f32,
    height: f32,
    color: Color,
    title: Title,
    rotation: f32,
    block_id: Entity,
}

#[derive(Component, Clone)]
struct Paper {
    width: f32,
    height: f32,
    color: Color,
    title: Title,
    rotation: f32,
    paper_id: Entity,
}

impl Paper {
    fn spawn(&self, commands: &mut Commands, position: Vec3) {
        let paper_entity = commands
            .spawn((
                self.clone(),
                SpriteBundle {
                    sprite: Sprite {
                        color: self.color,
                        custom_size: Some(Vec2::new(self.width, self.height)),
                        ..default()
                    },
                    transform: Transform::from_translation(position)
                        .with_rotation(Quat::from_rotation_z(self.rotation)),
                    ..default()
                },
            ))
            .id();

        commands
            .spawn((Text2dBundle {
                text: Text {
                    sections: vec![TextSection::new(
                        self.title.text.clone(),
                        TextStyle {
                            font_size: 32.0,
                            color: Color::BLACK,
                            ..default()
                        },
                    )],
                    justify: JustifyText::Center,
                    linebreak_behavior: BreakLineOn::WordBoundary,
                },
                text_2d_bounds: bevy::text::Text2dBounds {
                    size: Vec2::new(self.title.max_width, f32::INFINITY),
                },
                transform: Transform::from_xyz(self.title.position.x, self.title.position.y, 1.0),
                text_anchor: bevy::sprite::Anchor::Center,
                ..default()
            },))
            .set_parent(paper_entity);
    }
}

#[derive(Component)]
struct Word {
    text: String,
    font_size: f32,
    color: Color,
}

#[derive(Resource)]
struct CameraZoom {
    current: f32,
    target: f32,
    velocity: f32,
}

#[derive(Resource)]
struct CameraPosition {
    current: Vec2,
    target: Vec2,
    velocity: Vec2,
}

fn generate_random_address() -> String {
    let mut rng = rand::thread_rng();
    let hex_chars: Vec<char> = "0123456789abcdef".chars().collect();
    let mut address = String::from("0x");
    for _ in 0..40 {
        address.push(hex_chars[rng.gen_range(0..hex_chars.len())]);
    }
    address
}

fn generate_random_transaction() -> String {
    let actions = [
        "Transfer", "Swap", "Stake", "Unstake", "Mint", "Burn", "Delegate",
    ];
    let amounts = [
        "0.5 ETH",
        "100 USDC",
        "1000 DAI",
        "50 LINK",
        "25 UNI",
        "10 WBTC",
        "1000 USDT",
    ];
    let statuses = ["✓", "⏳", "❌"];

    let mut rng = rand::thread_rng();
    let action = actions[rng.gen_range(0..actions.len())];
    let amount = amounts[rng.gen_range(0..amounts.len())];
    let full_address = generate_random_address();
    let truncated_address = format!("0x{}...{}", &full_address[2..6], &full_address[38..42]);
    let status = statuses[rng.gen_range(0..statuses.len())];

    format!("{} {} to {} {}", action, amount, truncated_address, status)
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_plugins(DefaultPickingPlugins)
        .insert_resource(CameraZoom {
            current: 1.5,
            target: 1.5,
            velocity: 0.0,
        })
        .insert_resource(CameraPosition {
            current: Vec2::ZERO,
            target: Vec2::ZERO,
            velocity: Vec2::ZERO,
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (camera_ui_and_clicks, apply_camera_movement_and_zoom),
        )
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>, zoom: Res<CameraZoom>) {
    // Camera
    commands.spawn((Camera2dBundle {
        transform: Transform::from_scale(Vec3::splat(zoom.current)),
        ..default()
    },));

    let mut rng = rand::thread_rng();
    let block_count = 20; // Increased to 20 blocks
    let block_spacing = 600.0; // Vertical spacing between blocks
    let tx_spacing = 450.0; // Horizontal spacing between transactions
    let start_y = (block_count as f32 - 1.0) * block_spacing / 2.0; // Center vertically
    let leftmost_x = -2000.0; // Leftmost position for block headers

    // Create blocks vertically
    for block_idx in 0..block_count {
        let block_y = start_y - (block_idx as f32 * block_spacing);
        let txs_per_block = rng.gen_range(10..=50); // Random number of transactions per block

        // Create block header (blue post-it)
        let block_header = Paper {
            width: 400.0,
            height: 400.0,
            color: Color::rgb(0.3, 0.5, 1.0), // Blue color for block headers
            title: Title {
                text: format!("Block #{}", block_idx),
                font_size: 32.0,
                color: Color::WHITE,
                position: Vec2::new(0.0, 100.0),
                max_width: 340.0,
            },
            rotation: rng.gen_range(-0.1..0.1),
            paper_id: Entity::PLACEHOLDER, // Will be set after block creation
        };

        // Spawn block header at leftmost position
        block_header.spawn(&mut commands, Vec3::new(leftmost_x, block_y, 0.0));

        // Create transactions horizontally for this block, starting from left
        let start_x = leftmost_x + 500.0; // Start transactions after the block header
        for tx_idx in 0..txs_per_block {
            let paper = Paper {
                width: 400.0,
                height: 400.0,
                color: Color::rgb(0.9, 0.95, 1.0), // Light blue post-it
                title: Title {
                    text: generate_random_transaction(),
                    font_size: 28.0,
                    color: Color::rgb(0.1, 0.2, 0.3),
                    position: Vec2::new(0.0, 100.0),
                    max_width: 340.0,
                },
                rotation: rng.gen_range(-0.1..0.1),
                paper_id: Entity::PLACEHOLDER, // Will be set after block creation
            };
            paper.spawn(
                &mut commands,
                Vec3::new(start_x + (tx_idx as f32 * tx_spacing), block_y, 0.0),
            );
        }
    }
}

// --- UI and click handling system ---
fn camera_ui_and_clicks(
    mut zoom: ResMut<CameraZoom>,
    mut position: ResMut<CameraPosition>,
    mut contexts: EguiContexts,
    mut mouse_button_events: EventReader<MouseButtonInput>,
    paper_query: Query<&Transform, With<Paper>>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    windows: Query<&Window>,
) {
    // Handle zoom buttons and D-pad
    egui::Window::new("Camera Controls").show(contexts.ctx_mut(), |ui| {
        ui.horizontal(|ui| {
            if ui.button("Zoom In").clicked() {
                zoom.target /= 2.0;
            }
            if ui.button("Zoom Out").clicked() {
                zoom.target *= 2.0;
            }
        });
        ui.separator();
        ui.vertical_centered(|ui| {
            ui.horizontal(|ui| {
                ui.add_space(50.0);
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.add_space(25.0);
                        if ui
                            .add(egui::Button::new("UP").min_size(egui::vec2(50.0, 50.0)))
                            .clicked()
                        {
                            position.target.y += 100.0;
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui
                            .add(egui::Button::new("LE").min_size(egui::vec2(50.0, 50.0)))
                            .clicked()
                        {
                            position.target.x -= 100.0;
                        }
                        if ui
                            .add(egui::Button::new("RI").min_size(egui::vec2(50.0, 50.0)))
                            .clicked()
                        {
                            position.target.x += 100.0;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.add_space(25.0);
                        if ui
                            .add(egui::Button::new("DN").min_size(egui::vec2(50.0, 50.0)))
                            .clicked()
                        {
                            position.target.y -= 100.0;
                        }
                    });
                });
                ui.add_space(50.0);
            });
        });
    });

    // Handle paper clicks
    for event in mouse_button_events.read() {
        if event.button == MouseButton::Left && event.state.is_pressed() {
            let window = windows.single();
            let (camera, camera_transform) = camera_query.single();
            if let Some(mouse_pos) = window.cursor_position() {
                if let Some(world_pos) = camera.viewport_to_world(camera_transform, mouse_pos) {
                    for paper_transform in paper_query.iter() {
                        let paper_pos = paper_transform.translation.truncate();
                        let distance = paper_pos.distance(world_pos.origin.truncate());
                        if distance < 200.0 {
                            position.target = paper_pos;
                            break;
                        }
                    }
                }
            }
        }
    }
}

// --- Camera movement/zoom system ---
fn apply_camera_movement_and_zoom(
    time: Res<Time>,
    mut zoom: ResMut<CameraZoom>,
    mut position: ResMut<CameraPosition>,
    mut camera_query: Query<&mut Transform, With<Camera>>,
) {
    // Spring-based zoom
    let zoom_stiffness = 200.0;
    let zoom_damping = 0.8 * (zoom_stiffness as f32).sqrt();
    let dt = time.delta_seconds();

    let zoom_displacement = zoom.target - zoom.current;
    let zoom_spring_force = zoom_displacement * zoom_stiffness;
    let zoom_damping_force = -zoom.velocity * zoom_damping;
    let zoom_acceleration = zoom_spring_force + zoom_damping_force;

    zoom.velocity += zoom_acceleration * dt;
    zoom.current += zoom.velocity * dt;

    // Dramatic, punchy spring camera movement
    let stiffness = 200.0;
    let damping = 1.0 * (stiffness as f32).sqrt();
    let target = position.target;
    let current = position.current;
    let velocity = position.velocity;
    let displacement = target - current;
    let spring_force = displacement * stiffness;
    let damping_force = -velocity * damping;
    let acceleration = spring_force + damping_force;
    let new_velocity = velocity + acceleration * dt;
    let new_position = current + new_velocity * dt;
    position.velocity = new_velocity;
    position.current = new_position;

    // Apply to camera
    for mut transform in &mut camera_query {
        transform.translation.x = new_position.x;
        transform.translation.y = new_position.y;
        transform.scale = Vec3::splat(zoom.current);
    }
}
