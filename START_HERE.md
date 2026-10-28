# Start here

1. Install Python 3.12.
2. Double-click **setup_windows.bat** once; wait for it to finish.
3. Double-click **start_windows.bat** and leave the terminal open.
4. Open **http://127.0.0.1:8000** in Chrome.

The trained model is included. Select **Load examples → Analyze comments** to try it immediately.

For the extension: open **chrome://extensions**, enable **Developer mode**, choose **Load unpacked**, and select the **extension** folder. Open a YouTube video, scroll to its comments, then use Mozhi's **Analyze loaded comments** button.

For model improvement: use **Dataset studio → Load research dataset**, run **Label review → Find suspicious labels**, review a small batch yourself, then use **Experiments → Train & compare**.

The initial model is a measured research baseline. Published labels remain unverified, and the model misses substantial abuse. Scores are offensive-language likelihood, not abuse severity.

See **README.md** for setup details, data format, training, test commands, and limitations. **MODEL_CARD.md** contains the actual measured results.
