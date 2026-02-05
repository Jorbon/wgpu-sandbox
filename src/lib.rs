#![allow(dead_code)]

pub mod util;
pub mod app_state;
pub mod window_state;

use glyphon::{FontSystem, fontdb};
pub use util::*;
pub use app_state::*;
pub use window_state::*;


use std::{mem::size_of, sync::Arc};
use winit::{application::ApplicationHandler, dpi::{PhysicalPosition, PhysicalSize}, event::{DeviceEvent, DeviceId, KeyEvent, MouseButton, WindowEvent}, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowId}};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;



#[cfg(target_arch = "wasm32")]
pub mod canvas {
    use wasm_bindgen::UnwrapThrowExt;
    use wasm_bindgen::JsCast;
    
    const CANVAS_ID: &str = "canvas";

    pub fn get_canvas() -> web_sys::HtmlCanvasElement {
        let window = web_sys::window().expect_throw("Could not get window");
        let document = window.document().expect_throw("Could not get document");
        let canvas = document.get_element_by_id(CANVAS_ID).expect_throw(&format!("Could not get canvas with id: '{CANVAS_ID}'"));
        canvas.unchecked_into()
    }
    
    pub fn reset_canvas() -> web_sys::HtmlCanvasElement {
        let window = web_sys::window().expect_throw("Could not get window");
        let document = window.document().expect_throw("Could not get document");
        let canvas = document.get_element_by_id(CANVAS_ID).expect_throw(&format!("Could not get canvas with id: '{CANVAS_ID}'"));
        let new_canvas = document.create_element("canvas").expect_throw("Could not create new canvas");
        new_canvas.set_id(CANVAS_ID);
        canvas.parent_node().unwrap().replace_child(&new_canvas, &canvas).unwrap();
        canvas.remove();
        new_canvas.unchecked_into()
    }
}



pub struct App {
    pub app_state: AppState,
    pub window_state: Option<WindowState>,
    pub font_system: FontSystem,
    #[cfg(target_arch = "wasm32")] proxy: winit::event_loop::EventLoopProxy<WindowState>,
}

impl App {
    pub fn new(
        #[cfg(target_arch = "wasm32")] event_loop: &EventLoop<WindowState>,
    ) -> Self {
        
        let mut db = fontdb::Database::new();
        db.load_font_data(include_bytes!("../assets/fonts/Luciole-Regular.ttf").to_vec());
        // db.load_fonts_dir("assets/fonts");
        let mut font_system = FontSystem::new_with_locale_and_db(String::from("en-US"), db);
        
        Self {
            app_state: AppState::new(&mut font_system),
            window_state: None,
            font_system,
            #[cfg(target_arch = "wasm32")] proxy: event_loop.create_proxy(),
        }
    }
    
    pub fn create_window_state(&mut self, event_loop: &ActiveEventLoop) {
        #[cfg(not(target_arch = "wasm32"))] {
            let window = if let Some(window_state) = &self.window_state {
                if window_state.adapter.get_info().backend == wgpu::Backend::Dx12 && self.app_state.graphics_options.backend != Some(wgpu::Backend::Dx12) {
                    let mut attributes = Window::default_attributes()
                        .with_active(true)
                        .with_decorations(window_state.window.is_decorated())
                        .with_enabled_buttons(window_state.window.enabled_buttons())
                        .with_fullscreen(window_state.window.fullscreen())
                        .with_inner_size(window_state.window.inner_size())
                        .with_maximized(window_state.window.is_maximized())
                        .with_position(window_state.window.outer_position().unwrap())
                        .with_resizable(window_state.window.is_resizable())
                        .with_theme(window_state.window.theme())
                        .with_title(window_state.window.title())
                        .with_visible(window_state.window.is_visible().unwrap_or(true))
                        // .with_blur(blur)
                        // .with_content_protected(protected)
                        // .with_cursor(cursor)
                        // .with_max_inner_size(max_size)
                        // .with_min_inner_size(min_size)
                        // .with_transparent(transparent)
                        // .with_window_icon(window_icon)
                        // .with_window_level(level)
                        ;
                    
                    attributes.position = window_state.window.outer_position().ok().map(|position| winit::dpi::Position::Physical(position));
                    attributes.resize_increments = window_state.window.resize_increments().map(|size| winit::dpi::Size::Physical(size));
                    
                    Arc::new(event_loop.create_window(attributes).unwrap())
                } else {
                    Arc::clone(&window_state.window)
                }
            } else {
                let attributes = Window::default_attributes()
                    .with_inner_size(winit::dpi::LogicalSize::new(960.0, 720.0));
                    //.with_fullscreen(Some(winit::window::Fullscreen::Borderless(Some(event_loop.available_monitors().next().unwrap()))));
                Arc::new(event_loop.create_window(attributes).unwrap())
            };
            drop(self.window_state.take());
            self.on_new_window_state(pollster::block_on(WindowState::new(window, self.app_state.graphics_options)).unwrap());
        }
        #[cfg(target_arch = "wasm32")] {
            drop(self.window_state.take());
            
            use winit::platform::web::WindowAttributesExtWebSys;
            let window = Arc::new(event_loop.create_window(
                Window::default_attributes()
                    .with_canvas(Some(canvas::reset_canvas()))
            ).unwrap());
            
            let proxy = self.proxy.clone();
            let graphics_options = self.app_state.graphics_options;
            wasm_bindgen_futures::spawn_local(async move {
                assert!(proxy.send_event(WindowState::new(window, graphics_options).await.expect("Unable to set up canvas.")).is_ok());
            })
        }
    }
    
    pub fn on_new_window_state(&mut self, window_state: WindowState) {
        drop(self.window_state.take()); // Might still have existing version of state
        window_state.window.request_redraw();
        self.app_state.update_for_window_state(&window_state);
        let size = window_state.window.inner_size();
        self.window_state = Some(window_state);
        self.resize(size); // Rebuild resources
    }
    
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.app_state.on_resize(new_size);
        self.window_state.as_mut().map(|s| s.on_resize(new_size));
    }
    
    pub fn render(&mut self) {
        let window_state = match &mut self.window_state { Some(state) => state, None => return };
        
        let width = window_state.config.width;
        let height = window_state.config.height;
        
        self.app_state.on_frame(&mut self.font_system, width, height);
        
        window_state.queue.write_buffer(&window_state.uniform_buffer, 0, bytemuck::cast_slice(&[VertexUniforms {
            camera_transform: self.app_state.camera.get_transform(),
        }]));
        
        
        // Prepare menu
        let scale_factor = window_state.window.scale_factor() as f32;
        let resolution = window_state.menu_render_state.text_viewport.resolution();
        let logical_size = Vector([width as f32 / scale_factor, height as f32 / scale_factor]);
        
        if width != resolution.width || height != resolution.height {
            window_state.menu_render_state.text_viewport.update(&window_state.queue, glyphon::Resolution { width, height });
            window_state.queue.write_buffer(&window_state.menu_render_state.uniform_buffer, 0, bytemuck::cast_slice(&[MenuUniforms {
                scale_factor: Vector([1.0 / logical_size.x(), 1.0 / logical_size.y()]),
            }]));
            window_state.menu_render_state.layout_needs_update = true;
        }
        
        if window_state.menu_render_state.layout_needs_update {
            let box_areas = self.app_state.layout_menu(window_state, &mut self.font_system);
            
            if box_areas.len() != window_state.menu_render_state.box_area_cache.len() {
                window_state.menu_render_state.box_area_buffer = Some(window_state.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Menu box area buffer"),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    size: (box_areas.len() * size_of::<BoxArea>()) as wgpu::BufferAddress,
                    mapped_at_creation: false,
                }));
            }
            
            window_state.menu_render_state.box_area_cache = box_areas;
            
            let stride = size_of::<BoxAreaInstance>();
            if let Some(nz_length) = std::num::NonZero::new((window_state.menu_render_state.box_area_cache.len() * stride) as wgpu::BufferAddress) {
                if let Some(mut buf) = window_state.queue.write_buffer_with(window_state.menu_render_state.box_area_buffer.as_ref().unwrap(), 0, nz_length) {
                    let buf = &mut *buf;
                    for (i, box_area) in window_state.menu_render_state.box_area_cache.iter().enumerate() {
                        let offset = i * stride;
                        buf[offset..(offset + stride)].copy_from_slice(bytemuck::cast_slice(&[BoxAreaInstance {
                            rect: box_area.rect,
                            color: box_area.color,
                        }]));
                    }
                }
            }
            
            window_state.menu_render_state.layout_needs_update = false;
        }
        
        
        window_state.menu_render_state.text_renderer.prepare(
            &window_state.device,
            &window_state.queue,
            &mut self.font_system,
            &mut window_state.menu_render_state.atlas,
            &window_state.menu_render_state.text_viewport,
            self.app_state.menu.iter_text_areas().map(|text| text.get_render_object(scale_factor)),
            &mut window_state.menu_render_state.swash_cache
        ).unwrap();
        
        match window_state.render(&mut self.app_state) {
            Ok(_) => (),
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let size = window_state.window.inner_size();
                self.resize(size);
            }
            Err(e) => log::error!("Render broke uh oh: {e}")
        }
    }
}



impl ApplicationHandler<WindowState> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.create_window_state(event_loop);
        event_loop.set_control_flow(ControlFlow::Poll);
    }
    
    #[cfg(target_arch = "wasm32")]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: WindowState) {
        self.on_new_window_state(event);
    }
    
    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        // let window_state = match &mut self.window_state { Some(state) => state, None => return };
        
        match event {
            DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                if self.app_state.cursor_grab {
                    self.app_state.mouse_position.x += dx;
                    self.app_state.mouse_position.y += dy;
                }
            }
            _ => ()
        }
    }
    
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let window_state = match &mut self.window_state { Some(state) => state, None => return };
        
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.resize(size),
            
            WindowEvent::RedrawRequested => {
                window_state.window.request_redraw();
                self.render();
            }
            
            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(code), state: s, .. }, ..
            } => match (code, s.is_pressed()) {
                (KeyCode::Escape, true) => (),
                (KeyCode::KeyW      , pressed) => self.app_state.keys.w       = pressed,
                (KeyCode::KeyS      , pressed) => self.app_state.keys.s       = pressed,
                (KeyCode::KeyA      , pressed) => self.app_state.keys.a       = pressed,
                (KeyCode::KeyD      , pressed) => self.app_state.keys.d       = pressed,
                (KeyCode::Space     , pressed) => self.app_state.keys.space   = pressed,
                (KeyCode::ShiftLeft , pressed) => self.app_state.keys.shift   = pressed,
                (KeyCode::Digit0, true) => { window_state.config.desired_maximum_frame_latency = 0; window_state.surface.configure(&window_state.device, &window_state.config); }
                (KeyCode::Digit1, true) => { window_state.config.desired_maximum_frame_latency = 1; window_state.surface.configure(&window_state.device, &window_state.config); }
                (KeyCode::Digit2, true) => { window_state.config.desired_maximum_frame_latency = 2; window_state.surface.configure(&window_state.device, &window_state.config); }
                (KeyCode::Digit3, true) => { window_state.config.desired_maximum_frame_latency = 3; window_state.surface.configure(&window_state.device, &window_state.config); }
                (KeyCode::Digit4, true) => { window_state.config.desired_maximum_frame_latency = 4; window_state.surface.configure(&window_state.device, &window_state.config); }
                (KeyCode::KeyT, true) => {
                    println!("{:?}", self.app_state.camera.get_transform() * Vector([0.0f32, 0.0, 0.0, 1.0]).as_column());
                }
                _ => ()
            }
            
            WindowEvent::MouseInput { button, state, device_id: _ } => {
                let window_state = match &mut self.window_state { Some(state) => state, None => return };
                if let Some(id) = window_state.menu_render_state.find_box_at(self.app_state.mouse_position.to_logical(window_state.window.scale_factor())) {
                    if button == MouseButton::Left && state.is_pressed() {
                        
                        let mut graphics_options_changed = false;
                        match id {
                            MenuID::Graphics(id) => match id {
                                GraphicsMenuID::Backend(backend) => {
                                    if self.app_state.graphics_options.backend != Some(backend) { graphics_options_changed = true; }
                                    self.app_state.graphics_options.backend = Some(backend);
                                }
                                GraphicsMenuID::PowerPreference(preference) => {
                                    if self.app_state.graphics_options.power_preference != preference { graphics_options_changed = true; }
                                    self.app_state.graphics_options.power_preference = preference;
                                }
                                GraphicsMenuID::PresentMode(mode) => {
                                    if self.app_state.graphics_options.present_mode != mode { graphics_options_changed = true; }
                                    self.app_state.graphics_options.present_mode = mode;
                                }
                                GraphicsMenuID::SurfaceFormat(format) => {
                                    if self.app_state.graphics_options.surface_format != Some(format) { graphics_options_changed = true; }
                                    self.app_state.graphics_options.surface_format = Some(format);
                                }
                                GraphicsMenuID::AlphaMode(mode) => {
                                    if self.app_state.graphics_options.alpha_mode != Some(mode) { graphics_options_changed = true; }
                                    self.app_state.graphics_options.alpha_mode = Some(mode);
                                }
                            }
                            MenuID::Block => (),
                            MenuID::Pass => (),
                        }
                        
                        if graphics_options_changed {
                            self.create_window_state(event_loop);
                        }
                    }
                }
                
                
                // match (button, s.is_pressed()) {
                    // (MouseButton::Left, true) => {
                        // if !state.cursor_grab {
                        //     state.cursor_grab = true;
                        //     // state.window.set_cursor_grab(winit::window::CursorGrabMode::Locked).unwrap();
                        //     // state.window.set_cursor_grab(winit::window::CursorGrabMode::Confined).unwrap();
                        //     // state.window.set_cursor_visible(false);
                        // }
                    // }
                    
                    // _ => ()
                // }
            }
            
            WindowEvent::CursorMoved { position, device_id: _ } => {
                self.app_state.mouse_position = position;
            }
            
            _ => ()
        }
    }
}


#[cfg(not(target_arch = "wasm32"))]
pub fn run() -> Result<()> {
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Error).init();
    println!("desktop app started");
    
    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new();
    
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run() -> std::result::Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).unwrap_throw();
    log::info!("wasm app started");
    
    let event_loop = EventLoop::with_user_event().build().unwrap_throw();
    let app = App::new(&event_loop);
    
    // run_app works on wasm, but winit does something goofy with exceptions in it to keep the same return signature.
    // spawn_app does basically the same thing, but without this silliness, so the JS caller returns gracefully.
    use winit::platform::web::EventLoopExtWebSys;
    event_loop.spawn_app(app);
    Ok(())
}



