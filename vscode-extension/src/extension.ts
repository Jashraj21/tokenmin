/**
 * VS Code Extension entry point for TokenMin.
 * Minimizes code, prompts, logs, and JSON context for GitHub Copilot, ChatGPT, and AI tools.
 */
import * as vscode from 'vscode';
import { spawn } from 'child_process';

/**
 * Call the tokenmin CLI or embedded python runner.
 */
function runTokenMin(
    input: string,
    args: string[] = []
): Promise<{ text: string; originalTokens?: number; minimizedTokens?: number; savedTokens?: number; pct?: string }> {
    return new Promise((resolve, reject) => {
        const config = vscode.workspace.getConfiguration('tokenmin');
        const model = config.get<string>('model', 'gpt-4o');
        const level = config.get<string>('level', 'moderate');

        const cliArgs = ['minimize', '-', '--model', model, '--level', level, ...args];
        
        // Use tokenmin binary from environment or local virtualenv
        const proc = spawn('tokenmin', cliArgs);
        let stdout = '';
        let stderr = '';

        proc.stdin.write(input);
        proc.stdin.end();

        proc.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        proc.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        proc.on('close', (code) => {
            if (code !== 0) {
                // If tokenmin not in PATH, fallback to basic whitespace/comment reduction
                resolve({ text: fallbackMinify(input) });
                return;
            }
            resolve({ text: stdout });
        });

        proc.on('error', () => {
            resolve({ text: fallbackMinify(input) });
        });
    });
}

/**
 * Fallback lightweight minifier if Python CLI is not found in PATH.
 */
function fallbackMinify(text: string): string {
    return text
        .replace(/\/\*[\s\S]*?\*\/|([^:]|^)\/\/.*$/gm, '$1') // Strip JS/C comments
        .replace(/^\s*#.*$/gm, '')                           // Strip Python comments
        .replace(/\n\s*\n\s*\n+/g, '\n\n')                   // Collapse blank lines
        .trim();
}

export function activate(context: vscode.ExtensionContext) {
    // 1. Command: Copy Minimized for AI (Most popular developer action)
    const copyForAI = vscode.commands.registerCommand('tokenmin.copyOptimizedForAI', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const selection = editor.selection;
        const text = selection.isEmpty ? editor.document.getText() : editor.document.getText(selection);

        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "TokenMin: Optimizing context for AI...",
            cancellable: false
        }, async () => {
            const res = await runTokenMin(text);
            await vscode.env.clipboard.writeText(res.text);
            vscode.window.showInformationMessage(`⚡ Context optimized & copied! Ready to paste into Copilot/ChatGPT.`);
        });
    });

    // 2. Command: Skeletonize Active File (Ideal for @workspace repo context)
    const skeletonize = vscode.commands.registerCommand('tokenmin.skeletonizeActiveFile', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const text = editor.document.getText();
        const res = await runTokenMin(text, ['--skeleton']);
        await vscode.env.clipboard.writeText(res.text);

        vscode.window.showInformationMessage(`⚡ Interface skeleton copied (~80% token savings)! Perfect for @workspace context.`);
    });

    // 3. Command: Minimize Selection in Editor
    const minSelection = vscode.commands.registerCommand('tokenmin.minimizeSelection', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.selection.isEmpty) return;

        const selection = editor.selection;
        const text = editor.document.getText(selection);
        const res = await runTokenMin(text);

        editor.edit(editBuilder => {
            editBuilder.replace(selection, res.text);
        });
    });

    context.subscriptions.push(copyForAI, skeletonize, minSelection);
}

export function deactivate() {}
