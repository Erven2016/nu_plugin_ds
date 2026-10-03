use nu_plugin::{MsgPackSerializer, serve_plugin};
use nu_plugin_ds::DsPlugin;

fn main() {
    serve_plugin(&DsPlugin, MsgPackSerializer);
}
