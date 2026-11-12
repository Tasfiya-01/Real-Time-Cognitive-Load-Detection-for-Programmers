const vscode = require('vscode');
const http = require('http');

const BACKEND = 'http://localhost:5000';
const POLL_MS = 5000;   // EEG playback advances one 5-s window every 5 s

function getCurrentCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return '';

    const document = editor.document;
    const selection = editor.selection;

    // Selected text, or the first 500 characters of the file
    const selectedText = document.getText(selection);
    if (selectedText && selectedText.trim().length > 0) {
        return selectedText;
    }

    const fullText = document.getText();
    // Do not analyse the backend file itself
    if (fullText.includes('Flask') || fullText.includes('import joblib')) {
        return '';
    }
    return fullText.substring(0, 2000);
}

function brainText(eeg) {
    // e.g. "🧠 Engaged 0.98" or "😌 Resting 0.12"
    if (!eeg) return '🧠 EEG: n/a';
    const icon = eeg.state === 'Engaged' ? '🧠' : '😌';
    return `${icon} ${eeg.state} ${eeg.p_engaged.toFixed(2)}`;
}

function getCodeHelp() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('Please open a code file first!');
        return;
    }
    let code = getCurrentCode();
    if (!code) {
        code = editor.document.lineAt(editor.selection.active.line).text;
    }

    vscode.window.showInformationMessage('Getting AI suggestion...');

    const encodedCode = encodeURIComponent(code);
    http.get(`${BACKEND}/code_suggest?code=${encodedCode}`, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
            try {
                const result = JSON.parse(data);
                const codeTag = result.load_level ? `[Code: ${result.load_level}] ` : '';
                const brainTag = result.eeg ? `[${result.eeg.state}] ` : '';
                vscode.window.showInformationMessage(`💡 ${brainTag}${codeTag}${result.suggestion}`);
            } catch (e) {
                vscode.window.showErrorMessage('Error getting suggestion');
            }
        });
    }).on('error', () => {
        vscode.window.showErrorMessage('Backend offline!');
    });
}

function activate(context) {
    vscode.window.showInformationMessage('🧠 Cognitive Load Monitor v2 Started (EEG playback + code complexity)');

    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 1000);
    statusBar.text = '⚪ Connecting...';
    statusBar.show();
    context.subscriptions.push(statusBar);

    const cmd = vscode.commands.registerCommand('cognitiveLoad.getHelp', () => getCodeHelp());
    context.subscriptions.push(cmd);
    statusBar.command = 'cognitiveLoad.getHelp';

    let suggestionShown = false;

    function checkLoad() {
        const code = getCurrentCode();
        const encodedCode = encodeURIComponent(code);

        http.get(`${BACKEND}/predict?code=${encodedCode}`, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    const result = JSON.parse(data);
                    const codeLevel = result.load_level;          // code complexity: Low / Medium / High
                    const complexity = result.complexity;
                    const eeg = result.eeg;                       // brain state from EEG playback

                    const cc = complexity ? ` (CC:${complexity.cyclomatic_complexity})` : '';
                    const codeIcon = codeLevel === 'High' ? '🔴' : codeLevel === 'Medium' ? '🟡' : '🟢';
                    statusBar.text = `${brainText(eeg)} | ${codeIcon} Code: ${codeLevel.toUpperCase()}${cc}`;

                    statusBar.tooltip = eeg
                        ? `Brain state from EEG playback (participant P05, unseen by the model)\n` +
                          `P(engaged) = ${eeg.p_engaged} | actual phase: ${eeg.true_phase} (${eeg.snippet})\n` +
                          `Window ${eeg.window + 1}/${eeg.n_windows}\nClick for AI code suggestion`
                        : 'Click for AI code suggestion';

                    // Warn only when BOTH the code is complex AND the brain is engaged
                    const engaged = eeg && eeg.state === 'Engaged';
                    if (codeLevel === 'High' && engaged) {
                        statusBar.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
                        if (!suggestionShown) {
                            suggestionShown = true;
                            vscode.window.showWarningMessage(
                                `⚠️ Complex code while you are highly engaged. ${result.suggestion}`,
                                'Get Code Help',
                                'Dismiss'
                            ).then(sel => {
                                if (sel === 'Get Code Help') getCodeHelp();
                                setTimeout(() => { suggestionShown = false; }, 15000);
                            });
                        }
                    } else if (codeLevel === 'High' || codeLevel === 'Medium') {
                        statusBar.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
                        suggestionShown = false;
                    } else {
                        statusBar.backgroundColor = undefined;
                        suggestionShown = false;
                    }
                } catch (e) {
                    statusBar.text = '⚪ Error';
                }
            });
        }).on('error', () => {
            statusBar.text = '⚪ Backend offline';
            statusBar.backgroundColor = undefined;
        });
    }

    checkLoad();
    const timer = setInterval(checkLoad, POLL_MS);
    context.subscriptions.push({ dispose: () => clearInterval(timer) });
}

function deactivate() {}
module.exports = { activate, deactivate };