import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
import { VertexAI } from '@google-cloud/vertexai';
import { FunctionDeclarationSchemaType } from '@google-cloud/vertexai/build/src/types/content';
import type {
  GenerateContentResult,
  GenerateContentResponse,
  Content,
  Part,
  GenerationConfig,
  Tool,
} from '@google-cloud/vertexai/build/src/types/content';
import * as dotenv from 'dotenv';
import { Prompt } from './prompt';

dotenv.config();

interface Message {
  role: string;
  content: string;
}

// Add function type definitions
interface FunctionCall {
  name: string;
  args: Record<string, any>;
}

// Add time-related functions
function getCurrentTime(): string {
  const utcTime = new Date(Date.now());
  const localTime = new Date();
  const localTz = Intl.DateTimeFormat().resolvedOptions().timeZone;

  return `UTC Time: ${utcTime.toISOString()}\nLocal Time (${localTz}): ${localTime.toLocaleString()}`;
}

function getCurrentDate(): string {
  const now = new Date();
  return now.toLocaleDateString();
}

// Add Fibonacci function implementation
function getFibonacci(n: number): number {
  if (n <= 1) return n;
  let a = 0,
    b = 1;
  for (let i = 2; i <= n; i++) {
    const temp = a + b;
    a = b;
    b = temp;
  }
  return b;
}

// Define available functions
const tools: Tool[] = [
  {
    function_declarations: [
      {
        name: 'get_current_time',
        description: 'Get the current time in UTC and local timezone',
        parameters: {
          type: FunctionDeclarationSchemaType.OBJECT,
          properties: {},
        },
      },
      {
        name: 'get_current_weather',
        description: 'Get the current weather in a given location',
        parameters: {
          type: FunctionDeclarationSchemaType.OBJECT,
          properties: {
            location: {
              type: FunctionDeclarationSchemaType.STRING,
              description: 'The city or location to get weather for',
            },
            unit: {
              type: FunctionDeclarationSchemaType.STRING,
              enum: ['celsius', 'fahrenheit'],
              description: 'Temperature unit',
            },
          },
          required: ['location'],
        },
      },
      {
        name: 'get_fibonacci',
        description: 'Get the nth Fibonacci number',
        parameters: {
          type: FunctionDeclarationSchemaType.OBJECT,
          properties: {
            n: {
              type: FunctionDeclarationSchemaType.NUMBER,
              description: 'The position in the Fibonacci sequence (0-based index)',
            },
          },
          required: ['n'],
        },
      },
    ],
  },
];

class ChatBot {
  private projectId: string;
  private model: any;
  private history: Message[];
  private context: Message[];
  private maxContext: number;
  private vertexai: VertexAI | null;
  private prompt: Prompt;

  constructor() {
    this.projectId = process.env.MY_GOOGLE_PROJECTID || '';
    this.model = null;
    this.vertexai = null;
    this.history = [];
    this.context = [];
    this.maxContext = 10;

    // Initialize prompt
    this.prompt = new Prompt();
  }

  async initialize(): Promise<boolean> {
    if (!this.projectId) {
      console.error('Error: Environment variable MY_GOOGLE_PROJECTID is not set');
      return false;
    }

    console.log(`Successfully loaded project ID: ${this.projectId}`);

    try {
      this.vertexai = new VertexAI({
        project: this.projectId,
        location: 'us-central1',
      });

      // Initialize Gemini model with tools
      this.model = this.vertexai.getGenerativeModel({
        //model: 'gemini-pro',
        model: 'gemini-2.0-flash-exp',
        tools: tools,
        generation_config: {
          temperature: 0.1,
          top_p: 0.95,
          max_output_tokens: 8192,
        },
      });

      return true;
    } catch (error) {
      console.error('\nError: Google Cloud credentials not found!');
      console.error('Please set up your credentials using one of these methods:');
      console.error('1. Run: gcloud auth application-default login');
      console.error(
        '2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to your service account key file'
      );
      console.error(
        '\nFor more information, visit: https://cloud.google.com/docs/authentication/external/set-up-adc'
      );
      return false;
    }
  }

  private createContents(): Array<Content> {
    return this.context.map((message) => ({
      role: message.role,
      parts: [{ text: message.content }],
    }));
  }

  private updateContext(message: Message): void {
    this.context.push(message);
    if (this.context.length > this.maxContext * 2) {
      this.context = this.context.slice(2);
    }
  }

  private async processFunctionCall(functionCall: any): Promise<string> {
    switch (functionCall.name) {
      case 'get_current_time':
        return getCurrentTime();
      case 'get_current_weather':
        // Simulate weather API call
        const location = functionCall.args.location;
        const unit = functionCall.args.unit || 'celsius';
        return `Weather in ${location} is sunny and 22°${unit === 'celsius' ? 'C' : 'F'}`;
      case 'get_fibonacci':
        const n = parseInt(functionCall.args.n);
        if (isNaN(n) || n < 0) {
          return 'Please provide a valid non-negative number';
        }
        const result = getFibonacci(n);
        return `The ${n}th Fibonacci number is ${result}`;
      default:
        throw new Error(`Unknown function: ${functionCall.name}`);
    }
  }

  async generateResponse(userInput: string): Promise<void> {
    const userMessage: Message = { role: 'user', content: userInput };
    this.history.push(userMessage);
    this.updateContext(userMessage);

    try {
      if (!this.model) {
        throw new Error('Model not initialized');
      }

      let responseText = '';

      const generationConfig: GenerationConfig = {
        temperature: 1.0,
        top_p: 0.95,
        max_output_tokens: 8192,
      };

      // print contents, tools
      //console.log("Contents: ", this.createContents());
      //console.log("Tools: ", tools);

      const response = await this.model.generateContent({
        contents: this.createContents(),
        generation_config: generationConfig,
        tools: tools,
      });

      const result = await response.response;
      process.stdout.write('Gemini: ');
      //console.log(JSON.stringify(result, null, 2));

      if (result.candidates[0].content?.parts?.[0]?.functionCall) {
        const functionCall = result.candidates[0].content.parts[0].functionCall;
        try {
          responseText = await this.processFunctionCall(functionCall);
          console.log(responseText);

          const assistantMessage: Message = {
            role: 'assistant',
            content: responseText,
          };
          this.history.push(assistantMessage);
          this.updateContext(assistantMessage);
          return;
        } catch (error: unknown) {
          if (error instanceof Error) {
            console.error(`Error processing function call: ${error.message}`);
          } else {
            console.error('An unknown error occurred while processing function call');
          }
        }
      }

      if (result.candidates[0].content?.parts?.[0]?.text) {
        responseText = result.candidates[0].content.parts[0].text;
        console.log(responseText);
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: responseText,
      };

      this.history.push(assistantMessage);
      this.updateContext(assistantMessage);
    } catch (error: any) {
      if (error.message?.includes('429')) {
        console.error('Error: Rate limit exceeded. Please wait a moment before trying again.');
        console.error(
          'For more information, visit: https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429'
        );
      } else {
        console.error(`Error occurred: ${error.message}`);
      }
    }
  }

  async chatLoop(): Promise<void> {
    console.log("\nChat started! (Type 'quit' to exit, 'context' to see current context)");

    while (true) {
      try {
        const userInput = await this.prompt.readLine();

        const trimmedInput = userInput.trim();

        if (trimmedInput.toLowerCase() === 'quit') {
          console.log('Goodbye!');
          this.prompt.close();
          break;
        }

        if (trimmedInput.toLowerCase() === 'context') {
          console.log('\nCurrent context window:');
          this.context.forEach((msg) => {
            console.log(`${msg.role}: ${msg.content}`);
          });
          continue;
        }

        if (!trimmedInput) {
          continue;
        }

        await this.generateResponse(trimmedInput);
      } catch (error) {
        console.error('\nGoodbye!');
        this.prompt.close();
        break;
      }
    }
  }
}

async function main() {
  const chatbot = new ChatBot();
  if (await chatbot.initialize()) {
    await chatbot.chatLoop();
  }
}

if (require.main === module) {
  main().catch(console.error);
}
