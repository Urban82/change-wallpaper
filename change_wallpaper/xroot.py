from PIL import Image
from Xlib import X, Xatom, display

from change_wallpaper.log import get_logger

log = get_logger(__name__)


def set_root_background(dpy: display.Display, img: Image.Image) -> None:
    """Set img as root window background, compatible with hsetroot/feh conventions."""
    screen = dpy.screen()
    root = screen.root
    depth = screen.root_depth
    if depth not in (24, 32):
        raise RuntimeError(f"Unsupported root depth {depth}")

    width, height = screen.width_in_pixels, screen.height_in_pixels
    if img.size != (width, height):
        canvas = Image.new("RGB", (width, height))
        canvas.paste(img, (0, 0))
        img = canvas
    data = img.convert("RGB").tobytes("raw", "BGRX")

    pixmap = root.create_pixmap(width, height, depth)
    gc = pixmap.create_gc(foreground=screen.black_pixel)

    # Split the upload to stay below the X server's maximum request size
    stride = width * 4
    max_bytes = dpy.display.info.max_request_length * 4 - 1024
    if hasattr(dpy.display, "big_request_length") and dpy.display.big_request_length:
        max_bytes = min(dpy.display.big_request_length * 4 - 1024, 16 * 1024 * 1024)
    rows = max(1, max_bytes // stride)
    for y in range(0, height, rows):
        n = min(rows, height - y)
        pixmap.put_image(gc, 0, y, width, n, X.ZPixmap, depth, 0, data[y * stride:(y + n) * stride])
    gc.free()

    xroot_atom = dpy.intern_atom("_XROOTPMAP_ID")
    eset_atom = dpy.intern_atom("ESETROOT_PMAP_ID")

    # Free the pixmap left by a previous setter (hsetroot/Esetroot convention)
    old = root.get_full_property(eset_atom, Xatom.PIXMAP)
    old_root = root.get_full_property(xroot_atom, Xatom.PIXMAP)
    if old and old_root and old.value and old_root.value and old.value[0] == old_root.value[0]:
        log.debug("Freeing previous root pixmap 0x%x", old.value[0])
        dpy.create_resource_object("pixmap", old.value[0]).kill_client()

    root.change_property(xroot_atom, Xatom.PIXMAP, 32, [pixmap.id])
    root.change_property(eset_atom, Xatom.PIXMAP, 32, [pixmap.id])
    root.change_attributes(background_pixmap=pixmap)
    root.clear_area(0, 0, width, height)
    dpy.set_close_down_mode(X.RetainPermanent)
    dpy.sync()
    log.info("Root background set (%dx%d, pixmap 0x%x)", width, height, pixmap.id)
