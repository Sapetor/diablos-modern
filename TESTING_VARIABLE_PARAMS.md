# Testing Variable Parameters - Step by Step

## Test 1: Keyboard Shortcut

1. Run: `python diablos_modern.py`
2. Press `Cmd+Shift+V` (or use View → Show/Hide Variable Editor)
3. **Expected:** Variable Editor panel appears at bottom
4. **If it doesn't work:** Use the View menu instead

## Test 2: Define Variable

1. In Variable Editor, type:
   ```python
   K = 10
   ```
2. Click **"Update Workspace"** button
3. **Expected:** Toast message "✓ Workspace updated (1 variables)"
4. Look for log: `Workspace updated with 1 variables: ['K']`

## Test 3: Use Variable in Block

1. Drag a **Gain** block onto canvas
2. Click the Gain block to select it
3. In Properties panel on the right, find "Gain:" field
4. Click in the Gain field
5. Delete `1.0`
6. Type `K`
7. **IMPORTANT:** Click somewhere else (like on the canvas) to lose focus
8. **Expected log:** `Property changed: gain0.gain = K`
9. **Also expected:** `Updated gain0.gain to K`

## Test 4: Run Simulation

1. Add Step and Scope blocks
2. Connect: Step → Gain → Scope
3. Click Run (F5)
4. **Look for these logs:**
   ```
   Block gain0: params before resolve = {'gain': 'K', ...}
   WorkspaceManager variables = {'K': 10}
   Block gain0: exec_params after resolve = {'gain': 10, ...}
   ```

## Troubleshooting

**If property doesn't save when clicking away:**
- Try pressing Tab instead of clicking
- Try pressing Enter
- Check if ANY log appears when you type

**If Variable Editor doesn't open:**
- Use View menu instead of keyboard
- Check if QShortcut is properly initialized
