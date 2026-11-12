const vscode = require('vscode');
const http = require('http');

function getCurrentCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return '';
    
    const document = editor.document;
    const selection = editor.selection;
    
    // Selected text অথবা পুরো file এর প্রথম 500 characters
    const selectedText = document.getText(selection);
    if (selectedText && selectedText.trim().length > 0) {
        return selectedText;
    }
    
    // পুরো document এর content নিন
   const fullText = document.getText();
// Only send if it looks like actual code, not the backend file
if (fullText.includes('Flask') || fullText.includes('import joblib')) {
    return '';
}
return fullText.substring(0, 500); // পুরো file পাঠাও
}

function getCodeHelp() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('Please open a Python file first!');
        return;
    }
    let code = getCurrentCode();
    
    if (!code) {
        code = editor 
            ? editor.document.lineAt(editor.selection.active.line).text 
            : 'unknown code';
    }

    vscode.window.showInformationMessage('Getting AI suggestion...');

    const encodedCode = encodeURIComponent(code);
    http.get(
        `http://localhost:5000/code_suggest?code=${encodedCode}`,
        (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    const result = JSON.parse(data);
                    vscode.window.showInformationMessage(
                        
 `💡 ${result.load_level ? '[' + result.load_level + '] ' : ''}${result.suggestion}`
                    );
                } catch(e) {
                    vscode.window.showErrorMessage('Error getting suggestion');
                }
            });
        }
    ).on('error', () => {
        vscode.window.showErrorMessage('Backend offline!');
    });
}

function activate(context) {
    vscode.window.showInformationMessage('🧠 Cognitive Load Monitor v2 Started!');

    const statusBar = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right, 1000
    );
    statusBar.text = '⚪ Load: Connecting...';
    statusBar.show();
    context.subscriptions.push(statusBar);

    const cmd = vscode.commands.registerCommand(
        'cognitiveLoad.getHelp',
        () => getCodeHelp()
    );
    context.subscriptions.push(cmd);

    statusBar.command = 'cognitiveLoad.getHelp';
    statusBar.tooltip = '🧠 Click for AI code suggestion!';

    let suggestionShown = false;

    function checkLoad() {
        // Get current code to send with request
        const code = getCurrentCode();
        const encodedCode = encodeURIComponent(code);
        
        http.get(
            `http://localhost:5000/predict?code=${encodedCode}`,
            (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    try {
                        const result = JSON.parse(data);
                        const load = result.load_level;
                        const complexity = result.complexity;

                        // Show complexity info in status bar
                        let complexityInfo = '';
                        if (complexity) {
                            complexityInfo = ` (CC:${complexity.cyclomatic_complexity})`;
                        }

                        if (load === 'High') {
                            statusBar.text = `🔴 Load: HIGH${complexityInfo} ← Click!`;
                            statusBar.backgroundColor =
                                new vscode.ThemeColor('statusBarItem.errorBackground');

                            if (!suggestionShown) {
                                suggestionShown = true;
                                vscode.window.showWarningMessage(
                                    `🧠 HIGH Load! ${result.suggestion}`,
                                    'Get Code Help',
                                    'Dismiss'
                                ).then(sel => {
                                    if (sel === 'Get Code Help') {
                                        getCodeHelp();
                                    }
                                    setTimeout(() => {
                                        suggestionShown = false;
                                    }, 15000);
                                });
                            }

                        } else if (load === 'Medium') {
                            statusBar.text = `🟡 Load: MEDIUM${complexityInfo} ← Click`;
                            statusBar.backgroundColor = undefined;
                            suggestionShown = false;

                        } else {
                            statusBar.text = `🟢 Load: LOW${complexityInfo}`;
                            statusBar.backgroundColor = undefined;
                            suggestionShown = false;
                        }
                    } catch(e) {
                        statusBar.text = '⚪ Load: Error';
                    }
                });
            }
        ).on('error', () => {
            statusBar.text = '⚪ Load: Offline';
        });
    }

    checkLoad();
    setInterval(checkLoad, 30000);
}

function deactivate() {}
module.exports = { activate, deactivate };