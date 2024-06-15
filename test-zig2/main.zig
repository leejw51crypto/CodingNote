const std = @import("std");
const c = @cImport({
    @cInclude("example.h");
});

pub fn main() void {
    c.hello_from_c();
    std.debug.print("Hello from Zig!\n", .{});
}
