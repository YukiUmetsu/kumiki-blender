# Open your model in Blender

[← Home](https://github.com/YukiUmetsu/kumiki-blender/wiki) · [Previous: first model](https://github.com/YukiUmetsu/kumiki-blender/wiki/Make-Your-First-3D-Model) · [Next: change your model](https://github.com/YukiUmetsu/kumiki-blender/wiki/Change-Your-First-Model)

You should now have a `source.blend` file from the first lesson. Here's how to look around without accidentally destroying your model.

## 1. Open the project

1. Open Blender normally from Applications / Start menu.
2. Choose **File → Open**.
3. Navigate to the repo's `.review/lattice-observation-tower/r01` folder.
4. Select **`source.blend`** and open it.

Alternatively, double-click `source.blend` if your computer associates `.blend` files with Blender. If you cannot see `.review`, it is a hidden folder: see the [first model guide](https://github.com/YukiUmetsu/kumiki-blender/wiki/Make-Your-First-3D-Model#step-4-find-the-results).

You may see a floor, lights and a camera around the tower. Those are the **studio** used to render preview pictures. The exported `model.glb` is the model without the studio floor or lights.

## 2. Learn the four most useful controls

Move your mouse over the large 3D **Viewport** (the main scene area):

| I want to… | Mouse / keyboard action |
| --- | --- |
| Rotate my view around the model | Hold the **middle mouse button** (wheel) and drag, or use the little axis gizmo in the top-right of the Viewport |
| Zoom in/out | Scroll the mouse wheel, or use the zoom icon on the right side of the Viewport |
| Move the view sideways or vertically (pan) | **Shift + middle mouse button** and drag, or use the hand icon |
| Center the view on a selected part | Select it and press **Numpad .** (Frame Selected), or use **View → Frame Selected** |

On a Mac trackpad, start with the on-screen **axis gizmo**, hand and magnifying-glass icons. Blender's **Edit → Preferences → Input** menu has navigation-emulation options if you want trackpad-friendly shortcuts.

**Don't panic:** rotating the view does **not** rotate the actual model. It changes only where you are looking from.

## 3. What are these Blender panels?

- **3D Viewport (center):** the object you're looking at.
- **Outliner (upper right):** a list of scene collections and objects. Think of it like folders and items.
- **Properties (lower right):** settings for the selected object, such as its materials or modifiers.
- **Timeline (bottom):** where animation frames appear. You can ignore it for now.

Try clicking a part of the tower in the Outliner. It highlights in the scene. Different named parts make larger models easier to inspect.

## 4. Look at materials instead of plain gray surfaces

Near the top-right of the Viewport are small round **Viewport Shading** buttons. Try **Material Preview** to see surface color and material behavior. **Solid** mode is faster and useful for studying shapes. **Rendered** mode uses scene lighting and may be slow.

The `preview.png` file is a separately rendered picture. It can look different from the interactive Viewport because of lighting, reflection setup and render settings.

## 5. Save your own editable copy

Choose **File → Save As** and save a separate file outside `.review/r01` if you plan to experiment with Blender controls. This keeps the generated revision intact. The generator will not automatically apply your manual changes to future builds.

## 6. What about `model.glb`?

You can open a new Blender scene and use **File → Import → glTF 2.0 (.glb/.gltf)**, selecting `model.glb`. This is a good way to check what a downstream app will receive. The GLB may lack the preview studio's lights and floor; that is intentional.

You do not need to master Blender's modeling tools yet. Next, [Change Your First Model](https://github.com/YukiUmetsu/kumiki-blender/wiki/Change-Your-First-Model) by editing two simple recipe values.
