use bevy::ecs::message::MessageReader;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;
use mlua::prelude::*;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai;
use std::collections::HashMap;
use std::sync::{mpsc, Mutex};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = r#"You are a visual effects script generator for a 3D game engine.
Given a spell or effect name, output ONLY a Lua script (no markdown fences, no explanation).

Available functions:

  spawn_sphere(x, y, z, r, g, b, radius) -> id
      Spawn a sphere at (x,y,z) with RGB color [0-1] and given radius.

  spawn_cube(x, y, z, r, g, b, size) -> id
      Spawn a cube at (x,y,z) with RGB color [0-1] and given edge size.

  move_to(id, x, y, z, duration)
      Animate entity to (x,y,z) over `duration` seconds.

  set_lifetime(id, seconds)
      Entity disappears after `seconds`.

  set_scale(id, sx, sy, sz)
      Set entity scale factors.

  set_emission(id, r, g, b, intensity)
      Add emissive glow. Higher intensity = brighter.

Available Lua builtins: math.random(), math.sin(), math.cos(), math.pi,
math.sqrt(), math.abs(), loops, variables, tostring, tonumber, pairs, ipairs.

Scene: centre is (0,0,0). Y is up. Camera looks at origin from roughly (-2, 5, 10).
Create visually interesting effects with multiple entities, varied sizes, colors, and animations.
Output raw Lua only – no markdown, no comments outside the script."#;

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[derive(Component)]
struct SpellEntity(#[allow(dead_code)] u32);

#[derive(Component)]
struct AnimateMovement {
    start: Vec3,
    target: Vec3,
    duration: f32,
    elapsed: f32,
}

#[derive(Component)]
struct Lifetime(f32);

#[derive(Component)]
struct InputDisplay;

#[derive(Component)]
struct StatusDisplay;

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
struct SpellInput(String);

#[derive(Resource)]
struct LlmChannel {
    tx: Mutex<mpsc::Sender<String>>,
    rx: Mutex<mpsc::Receiver<String>>,
}

#[derive(Resource, Default)]
struct PendingCommands(Vec<SpellCommand>);

#[derive(Resource, Default)]
struct CastingState(bool);

// ---------------------------------------------------------------------------
// Lua → Bevy command buffer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum SpellCommand {
    SpawnSphere {
        id: u32,
        pos: [f32; 3],
        color: [f32; 3],
        radius: f32,
    },
    SpawnCube {
        id: u32,
        pos: [f32; 3],
        color: [f32; 3],
        size: f32,
    },
    MoveTo {
        id: u32,
        target: [f32; 3],
        duration: f32,
    },
    SetLifetime {
        id: u32,
        seconds: f32,
    },
    SetScale {
        id: u32,
        scale: [f32; 3],
    },
    SetEmission {
        id: u32,
        color: [f32; 3],
        intensity: f32,
    },
}

// ---------------------------------------------------------------------------
// App entry
// ---------------------------------------------------------------------------

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Caster – Spell VFX".into(),
                ..default()
            }),
            ..default()
        }))
        .init_resource::<SpellInput>()
        .init_resource::<PendingCommands>()
        .init_resource::<CastingState>()
        .add_systems(Startup, (setup_scene, setup_ui, setup_llm_channel))
        .add_systems(
            Update,
            (
                handle_keyboard_input,
                poll_llm_response,
                process_spell_commands,
                animate_movement,
                tick_lifetimes,
            )
                .chain(),
        )
        .run();
}

// ---------------------------------------------------------------------------
// Startup systems
// ---------------------------------------------------------------------------

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(20.0, 20.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.15, 0.15, 0.18),
            ..default()
        })),
    ));

    // Light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            intensity: 2_000_000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 10.0, 4.0),
    ));

    // Second fill light for better visibility
    commands.spawn((
        PointLight {
            intensity: 800_000.0,
            ..default()
        },
        Transform::from_xyz(-6.0, 8.0, -4.0),
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn setup_ui(mut commands: Commands) {
    // Status text (top-left)
    commands.spawn((
        Text::new("Type a spell name and press Enter"),
        TextFont {
            font_size: 22.0,
            ..default()
        },
        TextColor(Color::srgb(0.8, 0.8, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(16.0),
            ..default()
        },
        StatusDisplay,
    ));

    // Input bar (bottom)
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Px(48.0),
                padding: UiRect::horizontal(Val::Px(16.0)),
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.08, 0.08, 0.10, 0.95)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("> _"),
                TextFont {
                    font_size: 22.0,
                    ..default()
                },
                TextColor(Color::srgb(0.2, 1.0, 0.4)),
                InputDisplay,
            ));
        });
}

fn setup_llm_channel(mut commands: Commands) {
    let (spell_tx, spell_rx) = mpsc::channel::<String>();
    let (script_tx, script_rx) = mpsc::channel::<String>();

    std::thread::spawn(move || {
        if std::env::var("OPENAI_API_KEY").is_err() {
            let _ = script_tx
                .send("-- ERROR: OPENAI_API_KEY environment variable not set".into());
            return;
        }

        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                let _ = script_tx.send(format!("-- ERROR: failed to create runtime: {e}"));
                return;
            }
        };

        let client = openai::Client::from_env();

        let agent = client.agent("gpt-4o").preamble(SYSTEM_PROMPT).build();

        while let Ok(spell_name) = spell_rx.recv() {
            let prompt_text = format!("Create a visual effect for: {spell_name}");
            let result = rt.block_on(async { agent.prompt(&prompt_text).await });
            match result {
                Ok(script) => {
                    let _ = script_tx.send(strip_code_blocks(&script));
                }
                Err(e) => {
                    let _ = script_tx.send(format!("-- ERROR: {e}"));
                }
            }
        }
    });

    commands.insert_resource(LlmChannel {
        tx: Mutex::new(spell_tx),
        rx: Mutex::new(script_rx),
    });
}

// ---------------------------------------------------------------------------
// Update systems
// ---------------------------------------------------------------------------

fn handle_keyboard_input(
    mut reader: MessageReader<KeyboardInput>,
    mut input: ResMut<SpellInput>,
    mut input_display: Query<&mut Text, With<InputDisplay>>,
    mut casting: ResMut<CastingState>,
    channel: Res<LlmChannel>,
    mut status: Query<&mut Text, (With<StatusDisplay>, Without<InputDisplay>)>,
    spell_entities: Query<Entity, With<SpellEntity>>,
    mut commands: Commands,
) {
    // Drain events even when casting so they don't pile up
    let events: Vec<_> = reader.read().cloned().collect();

    if casting.0 {
        return;
    }

    let mut changed = false;
    let mut submit = false;

    for event in &events {
        if !event.state.is_pressed() {
            continue;
        }
        match &event.logical_key {
            Key::Backspace => {
                input.0.pop();
                changed = true;
            }
            Key::Enter => {
                submit = true;
            }
            Key::Character(c) => {
                let s: &str = c.as_str();
                if s.chars().all(|ch| !ch.is_control()) {
                    input.0.push_str(s);
                    changed = true;
                }
            }
            _ => {}
        }
    }

    if changed && !submit {
        for mut text in &mut input_display {
            *text = Text::new(format!("> {}_", input.0));
        }
    }

    if submit && !input.0.trim().is_empty() {
        let spell_name = input.0.trim().to_string();

        // Clear previous spell entities
        for entity in &spell_entities {
            commands.entity(entity).despawn();
        }

        // Send to LLM
        if channel
            .tx
            .lock()
            .unwrap()
            .send(spell_name.clone())
            .is_ok()
        {
            casting.0 = true;
            for mut text in &mut status {
                *text = Text::new(format!("Casting '{spell_name}'..."));
            }
            for mut text in &mut input_display {
                *text = Text::new(format!("> {spell_name}"));
            }
        }
    }
}

fn poll_llm_response(
    channel: Res<LlmChannel>,
    mut casting: ResMut<CastingState>,
    mut pending: ResMut<PendingCommands>,
    mut status: Query<&mut Text, With<StatusDisplay>>,
    mut input: ResMut<SpellInput>,
    mut input_display: Query<&mut Text, (With<InputDisplay>, Without<StatusDisplay>)>,
) {
    if !casting.0 {
        return;
    }

    let script: String = match channel.rx.lock().unwrap().try_recv() {
        Ok(s) => s,
        Err(mpsc::TryRecvError::Empty) => return,
        Err(mpsc::TryRecvError::Disconnected) => {
            casting.0 = false;
            for mut text in &mut status {
                *text = Text::new("LLM thread disconnected");
            }
            return;
        }
    };

    casting.0 = false;
    input.0.clear();
    for mut text in &mut input_display {
        *text = Text::new("> _");
    }

    if script.starts_with("-- ERROR:") {
        for mut text in &mut status {
            *text = Text::new(script.trim_start_matches("-- ERROR:").trim().to_string());
        }
        return;
    }

    // Execute Lua script
    match execute_lua_script(&script) {
        Ok(cmds) => {
            let count = cmds.len();
            pending.0 = cmds;
            for mut text in &mut status {
                *text = Text::new(format!("Spell cast! ({count} commands)"));
            }
        }
        Err(e) => {
            for mut text in &mut status {
                *text = Text::new(format!("Lua error: {e}"));
            }
        }
    }
}

fn process_spell_commands(
    mut commands: Commands,
    mut pending: ResMut<PendingCommands>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if pending.0.is_empty() {
        return;
    }

    let spell_cmds = std::mem::take(&mut pending.0);
    let mut id_to_entity: HashMap<u32, Entity> = HashMap::new();
    let mut id_to_material: HashMap<u32, Handle<StandardMaterial>> = HashMap::new();
    let mut id_to_pos: HashMap<u32, Vec3> = HashMap::new();

    for cmd in spell_cmds {
        match cmd {
            SpellCommand::SpawnSphere {
                id,
                pos,
                color,
                radius,
            } => {
                let mat_handle = materials.add(StandardMaterial {
                    base_color: Color::srgb(color[0], color[1], color[2]),
                    ..default()
                });
                let position = Vec3::new(pos[0], pos[1], pos[2]);
                let entity = commands
                    .spawn((
                        Mesh3d(meshes.add(Sphere::new(radius))),
                        MeshMaterial3d(mat_handle.clone()),
                        Transform::from_translation(position),
                        SpellEntity(id),
                    ))
                    .id();
                id_to_entity.insert(id, entity);
                id_to_material.insert(id, mat_handle);
                id_to_pos.insert(id, position);
            }
            SpellCommand::SpawnCube {
                id,
                pos,
                color,
                size,
            } => {
                let mat_handle = materials.add(StandardMaterial {
                    base_color: Color::srgb(color[0], color[1], color[2]),
                    ..default()
                });
                let position = Vec3::new(pos[0], pos[1], pos[2]);
                let entity = commands
                    .spawn((
                        Mesh3d(meshes.add(Cuboid::new(size, size, size))),
                        MeshMaterial3d(mat_handle.clone()),
                        Transform::from_translation(position),
                        SpellEntity(id),
                    ))
                    .id();
                id_to_entity.insert(id, entity);
                id_to_material.insert(id, mat_handle);
                id_to_pos.insert(id, position);
            }
            SpellCommand::MoveTo {
                id,
                target,
                duration,
            } => {
                if let Some(&entity) = id_to_entity.get(&id) {
                    let start = id_to_pos.get(&id).copied().unwrap_or(Vec3::ZERO);
                    let target_vec = Vec3::new(target[0], target[1], target[2]);
                    commands.entity(entity).insert(AnimateMovement {
                        start,
                        target: target_vec,
                        duration: duration.max(0.01),
                        elapsed: 0.0,
                    });
                    id_to_pos.insert(id, target_vec);
                }
            }
            SpellCommand::SetLifetime { id, seconds } => {
                if let Some(&entity) = id_to_entity.get(&id) {
                    commands.entity(entity).insert(Lifetime(seconds));
                }
            }
            SpellCommand::SetScale { id, scale } => {
                if let Some(&entity) = id_to_entity.get(&id) {
                    let pos = id_to_pos.get(&id).copied().unwrap_or(Vec3::ZERO);
                    commands.entity(entity).insert(Transform {
                        translation: pos,
                        scale: Vec3::new(scale[0], scale[1], scale[2]),
                        ..default()
                    });
                }
            }
            SpellCommand::SetEmission {
                id,
                color,
                intensity,
            } => {
                if let Some(mat_handle) = id_to_material.get(&id) {
                    if let Some(mat) = materials.get_mut(mat_handle) {
                        mat.emissive = LinearRgba::new(
                            color[0] * intensity,
                            color[1] * intensity,
                            color[2] * intensity,
                            1.0,
                        );
                    }
                }
            }
        }
    }
}

fn animate_movement(
    time: Res<Time>,
    mut commands: Commands,
    mut query: Query<(Entity, &mut Transform, &mut AnimateMovement)>,
) {
    for (entity, mut transform, mut anim) in &mut query {
        anim.elapsed += time.delta_secs();
        let t = (anim.elapsed / anim.duration).clamp(0.0, 1.0);
        // Smooth ease-out interpolation
        let t_smooth = 1.0 - (1.0 - t).powi(3);
        transform.translation = anim.start.lerp(anim.target, t_smooth);
        if t >= 1.0 {
            commands.entity(entity).remove::<AnimateMovement>();
        }
    }
}

fn tick_lifetimes(
    time: Res<Time>,
    mut commands: Commands,
    mut query: Query<(Entity, &mut Lifetime)>,
) {
    for (entity, mut lifetime) in &mut query {
        lifetime.0 -= time.delta_secs();
        if lifetime.0 <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}

// ---------------------------------------------------------------------------
// Lua sandbox
// ---------------------------------------------------------------------------

fn execute_lua_script(script: &str) -> Result<Vec<SpellCommand>, String> {
    let lua = Lua::new_with(mlua::StdLib::ALL_SAFE, LuaOptions::default())
        .map_err(|e| e.to_string())?;

    let commands = std::sync::Arc::new(std::sync::Mutex::new(Vec::<SpellCommand>::new()));
    let next_id = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(1));

    // -- spawn_sphere(x, y, z, r, g, b, radius) -> id
    {
        let cmds = commands.clone();
        let ids = next_id.clone();
        let f = lua
            .create_function(
                move |_, (x, y, z, r, g, b, radius): (f32, f32, f32, f32, f32, f32, f32)| {
                    let id = ids.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    cmds.lock().unwrap().push(SpellCommand::SpawnSphere {
                        id,
                        pos: [x, y, z],
                        color: [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)],
                        radius: radius.max(0.01),
                    });
                    Ok(id)
                },
            )
            .map_err(|e| e.to_string())?;
        lua.globals()
            .set("spawn_sphere", f)
            .map_err(|e| e.to_string())?;
    }

    // -- spawn_cube(x, y, z, r, g, b, size) -> id
    {
        let cmds = commands.clone();
        let ids = next_id.clone();
        let f = lua
            .create_function(
                move |_, (x, y, z, r, g, b, size): (f32, f32, f32, f32, f32, f32, f32)| {
                    let id = ids.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    cmds.lock().unwrap().push(SpellCommand::SpawnCube {
                        id,
                        pos: [x, y, z],
                        color: [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)],
                        size: size.max(0.01),
                    });
                    Ok(id)
                },
            )
            .map_err(|e| e.to_string())?;
        lua.globals()
            .set("spawn_cube", f)
            .map_err(|e| e.to_string())?;
    }

    // -- move_to(id, x, y, z, duration)
    {
        let cmds = commands.clone();
        let f = lua
            .create_function(move |_, (id, x, y, z, dur): (u32, f32, f32, f32, f32)| {
                cmds.lock().unwrap().push(SpellCommand::MoveTo {
                    id,
                    target: [x, y, z],
                    duration: dur.max(0.01),
                });
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        lua.globals()
            .set("move_to", f)
            .map_err(|e| e.to_string())?;
    }

    // -- set_lifetime(id, seconds)
    {
        let cmds = commands.clone();
        let f = lua
            .create_function(move |_, (id, secs): (u32, f32)| {
                cmds.lock().unwrap().push(SpellCommand::SetLifetime {
                    id,
                    seconds: secs.max(0.0),
                });
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        lua.globals()
            .set("set_lifetime", f)
            .map_err(|e| e.to_string())?;
    }

    // -- set_scale(id, sx, sy, sz)
    {
        let cmds = commands.clone();
        let f = lua
            .create_function(move |_, (id, sx, sy, sz): (u32, f32, f32, f32)| {
                cmds.lock().unwrap().push(SpellCommand::SetScale {
                    id,
                    scale: [sx, sy, sz],
                });
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        lua.globals()
            .set("set_scale", f)
            .map_err(|e| e.to_string())?;
    }

    // -- set_emission(id, r, g, b, intensity)
    {
        let cmds = commands.clone();
        let f = lua
            .create_function(move |_, (id, r, g, b, intensity): (u32, f32, f32, f32, f32)| {
                cmds.lock().unwrap().push(SpellCommand::SetEmission {
                    id,
                    color: [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)],
                    intensity: intensity.max(0.0),
                });
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        lua.globals()
            .set("set_emission", f)
            .map_err(|e| e.to_string())?;
    }

    // Build sandboxed environment: whitelist only safe globals + our functions
    let env = lua.create_table().map_err(|e| e.to_string())?;
    let globals = lua.globals();
    for name in [
        "spawn_sphere",
        "spawn_cube",
        "move_to",
        "set_lifetime",
        "set_scale",
        "set_emission",
        "print",
        "math",
        "string",
        "table",
        "type",
        "tostring",
        "tonumber",
        "pairs",
        "ipairs",
        "select",
        "unpack",
        "pcall",
        "error",
    ] {
        if let Ok(val) = globals.get::<mlua::Value>(name.to_string()) {
            let _ = env.set(name, val);
        }
    }

    lua.load(script)
        .set_name("spell_script")
        .set_environment(env)
        .exec()
        .map_err(|e| e.to_string())?;

    let result = commands.lock().unwrap().clone();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn strip_code_blocks(s: &str) -> String {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("```") {
        let rest = rest.strip_prefix("lua").unwrap_or(rest);
        let rest = rest.strip_prefix('\n').unwrap_or(rest);
        if let Some(body) = rest.strip_suffix("```") {
            return body.trim().to_string();
        }
    }
    s.to_string()
}
