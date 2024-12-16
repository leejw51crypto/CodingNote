import * as readline from 'readline';

interface ReadlineInterfaceWithHistory extends readline.Interface {
  history: string[];
  historyIndex: number;
  line: string;
  cursor: number;
}

export class Prompt {
  private rl: ReadlineInterfaceWithHistory;
  private currentLine: string = '';
  private currentCursor: number = 0;

  constructor() {
    // Initialize readline interface with raw mode for key handling
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: true,
    }) as ReadlineInterfaceWithHistory;

    // Initialize history
    this.rl.history = [];
    this.rl.historyIndex = -1;

    // Enable raw mode for key handling
    process.stdin.setRawMode(true);
    process.stdin.on('keypress', this.handleKeypress.bind(this));
  }

  private handleKeypress(str: string, key: any): void {
    if (key.name === 'up') {
      // Navigate history up - show more recent commands first
      if (this.rl.historyIndex < this.rl.history.length - 1) {
        this.rl.historyIndex++;
        // Get from end of array - most recent commands are at the end
        this.setLine(this.rl.history[this.rl.history.length - this.rl.historyIndex]);
      }
    } else if (key.name === 'down') {
      // Navigate history down - show more recent commands
      if (this.rl.historyIndex > 0) {
        this.rl.historyIndex--;
        this.setLine(this.rl.history[this.rl.history.length - this.rl.historyIndex]);
      } else if (this.rl.historyIndex === 0) {
        // Clear line when reaching the end
        this.rl.historyIndex = -1;
        this.setLine('');
      }
    }
  }

  private setLine(line: string): void {
    // Clear current line
    readline.clearLine(process.stdout, 0);
    readline.cursorTo(process.stdout, 0);

    // Write prompt and new line
    process.stdout.write('You: ' + line);
    this.currentLine = line;
    this.currentCursor = line.length;
  }

  public async readLine(prompt: string = 'You: '): Promise<string> {
    return new Promise((resolve) => {
      this.rl.question(prompt, (answer) => {
        // Add non-empty answers to history at the end
        if (answer.trim()) {
          this.rl.history.push(answer);
        }
        this.rl.historyIndex = -1;
        resolve(answer);
      });
    });
  }

  public close(): void {
    this.rl.close();
    process.stdin.setRawMode(false);
  }
}

// Example usage:
async function test() {
  const prompt = new Prompt();

  try {
    while (true) {
      const input = await prompt.readLine();
      if (input.toLowerCase() === 'quit') {
        break;
      }
      console.log('Received:', input);
    }
  } finally {
    prompt.close();
  }
}

if (require.main === module) {
  test().catch(console.error);
}
