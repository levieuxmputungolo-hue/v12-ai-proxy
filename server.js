require('dotenv').config();
const express = require('express');
const cors = require('cors');
const OpenAI = require('openai');
const Groq = require('groq-sdk');

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

const PORT = process.env.PORT || 3001;

const OPENAI_KEY = process.env.OPENAI_API_KEY || '';
const GROQ_KEY = process.env.GROQ_API_KEY || '';
const TOGETHER_KEY = process.env.TOGETHER_API_KEY || '';
const HF_KEY = process.env.HF_API_KEY || '';
const V12_API_KEY = process.env.V12_API_KEY || 'v12-dev-key';

const openai = OPENAI_KEY ? new OpenAI({ apiKey: OPENAI_KEY }) : null;
const groq = GROQ_KEY ? new Groq({ apiKey: GROQ_KEY }) : null;
const together = TOGETHER_KEY ? new OpenAI({ apiKey: TOGETHER_KEY, baseURL: 'https://api.together.xyz/v1' }) : null;
const hf = HF_KEY ? new OpenAI({ apiKey: HF_KEY, baseURL: 'https://api-inference.huggingface.co/v1' }) : null;

// ═══════════════════════════════════════════════════════════
// SYSTEM PROMPT V12 AI — Structure Role/Consignes/Format
// ═══════════════════════════════════════════════════════════
const SYSTEM_PROMPT = `[ROLE ET IDENTITE]
Tu es V12 AI, un assistant virtuel avance developpe pour offrir une assistance precise en developpement, analyse de donnees et automatisation. Tu es intelligent, precis et serviable. Tu es supreme.

[CONSIGNES DE COMPORTEMENT]
1. Analyse toujours la demande de l'utilisateur avant de repondre.
2. Si une demande necessite la creation ou la manipulation d'un document (Excel, PDF) ou une recherche en ligne, utilise imperativement le Function Calling approprie.
3. Conserve un ton professionnel, clair et concis.
4. Reponds dans la langue utilisee par l'utilisateur (par defaut en francais).
5. Ne fabrique pas de faits ou d'informations. Si une donnee est inconnue ou hors de portee, indique-le clairement.
6. Tu connais la RDC, la culture congolaise, Kinshasa, et l'Afrique.
7. Tu es expert en programmation, sciences, philosophie, cuisine, musique.

[FORMAT DE SORTIE]
- Utilise le balisage Markdown pour structurer tes explications (titres, listes, blocs de code).
- Sois concis mais complet.
- Utilise des emojis avec moderation.
- Si tu utilises un outil, explique brievement ce que tu fais.

[CAPACITES TECHNIQUES]
- Tu peux creer des fichiers Excel (.xlsx) via create_excel
- Tu peux creer des fichiers PDF (.pdf) via create_pdf
- Tu peux rechercher sur le web via web_search
- Tu peux executer du JavaScript via run_code
- Tu peux analyser des donnees et faire des calculs precis
- Tu peux aider pour la cuisine, la musique, le dessin, la 3D`;

// ═══════════════════════════════════════════════════════════
// TOOL DEFINITIONS (OpenAI Function Calling format)
// ═══════════════════════════════════════════════════════════
const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'create_excel',
      description: 'Cree un fichier Excel (.xlsx) avec des donnees structurees. Utilise quand l\'utilisateur veut un tableau, classeur, export Excel, ou des donnees en format tableur.',
      parameters: {
        type: 'object',
        properties: {
          filename: { type: 'string', description: 'Nom du fichier sans extension (ex: rapport_ventes)' },
          data: {
            type: 'array',
            description: 'Donnees en tableau 2D. Premiere ligne = en-tetes, lignes suivantes = donnees.',
            items: { type: 'array', items: { type: 'string' } }
          }
        },
        required: ['filename', 'data']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'create_pdf',
      description: 'Cree un fichier PDF avec du contenu structure. Utilise quand l\'utilisateur veut un document, rapport, facture, lettre, ou resume PDF.',
      parameters: {
        type: 'object',
        properties: {
          filename: { type: 'string', description: 'Nom du fichier sans extension' },
          title: { type: 'string', description: 'Titre du document' },
          paragraphs: {
            type: 'array',
            items: { type: 'string' },
            description: 'Paragraphes du document (un par element du tableau)'
          },
          markdown: { type: 'string', description: 'Contenu en Markdown (alternative aux paragraphes)' }
        },
        required: ['filename']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'web_search',
      description: 'Recherche des informations sur internet. Utilise quand l\'utilisateur cherche des informations actuelles, des actualites, des prix, ou des donnees en temps reel.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Terme de recherche precis et detaille' },
          numResults: { type: 'number', description: 'Nombre de resultats souhaites (defaut: 5, max: 10)' }
        },
        required: ['query']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_code',
      description: 'Execute du code JavaScript dans un environnement securise. Utilise pour les calculs complexes, la transformation de donnees, ou tester du code.',
      parameters: {
        type: 'object',
        properties: {
          code: { type: 'string', description: 'Code JavaScript a executer' },
          language: { type: 'string', description: 'Langage (javascript par defaut)' }
        },
        required: ['code']
      }
    }
  }
];

// ═══════════════════════════════════════════════════════════
// CONTEXT WINDOW TRUNCATION
// ═══════════════════════════════════════════════════════════
const MAX_HISTORY_MESSAGES = 20;
const MAX_CONTEXT_TOKENS = 4000; // approx

function estimateTokens(text) {
  return Math.ceil(text.length / 4);
}

function truncateContext(messages) {
  if (messages.length <= MAX_HISTORY_MESSAGES) return messages;

  // Keep system prompt + last N messages
  const systemMsg = messages[0]; // system prompt
  const conversation = messages.slice(1);

  // Keep last MAX_HISTORY_MESSAGES
  const kept = conversation.slice(-MAX_HISTORY_MESSAGES);

  // Estimate total tokens
  let totalTokens = 0;
  const result = [systemMsg];

  for (let i = kept.length - 1; i >= 0; i--) {
    const msgTokens = estimateTokens(kept[i].content || '');
    if (totalTokens + msgTokens > MAX_CONTEXT_TOKENS) break;
    totalTokens += msgTokens;
    result.unshift(kept[i]);
  }

  return result;
}

// ═══════════════════════════════════════════════════════════
// WEB SEARCH
// ═══════════════════════════════════════════════════════════
async function webSearch(query, numResults = 5) {
  try {
    const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`;
    const res = await fetch(url);
    const data = await res.json();
    const results = [];
    if (data.AbstractText) {
      results.push({ title: data.Heading || query, snippet: data.AbstractText, url: data.AbstractURL || '' });
    }
    if (data.RelatedTopics) {
      for (const t of data.RelatedTopics) {
        if (t.Text && results.length < numResults) {
          results.push({ title: t.Text.split(' - ')[0] || '', snippet: t.Text, url: t.FirstURL || '' });
        }
      }
    }
    return { ok: true, query, results, total: results.length };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

// ═══════════════════════════════════════════════════════════
// RUN CODE (sandbox)
// ═══════════════════════════════════════════════════════════
async function runCode(code, language = 'javascript') {
  try {
    const logs = [];
    const fakeConsole = { log: (...args) => logs.push(args.join(' ')) };
    const fn = new Function('console', 'fetch', code);
    try {
      const result = fn(fakeConsole, fetch);
      return { ok: true, logs, result: String(result) };
    } catch (err) {
      return { ok: false, logs, error: err.message };
    }
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

// ═══════════════════════════════════════════════════════════
// AUTH MIDDLEWARE
// ═══════════════════════════════════════════════════════════
function authMiddleware(req, res, next) {
  const key = req.headers['x-api-key'] || req.query.key;
  if (V12_API_KEY && key !== V12_API_KEY) return res.status(401).json({ error: 'API key invalide' });
  next();
}

// ═══════════════════════════════════════════════════════════
// MAIN CHAT ENDPOINT
// ═══════════════════════════════════════════════════════════
app.get('/health', (req, res) => { res.json({ ok: true, status: 'alive', model: 'openai/gpt-oss-120b', timestamp: Date.now() }); });

app.post('/api/chat', async (req, res) => {
  try {
    const { messages, model, apiKey } = req.body;
    if (!messages || !Array.isArray(messages)) return res.status(400).json({ error: 'messages requis' });

    // Truncate context to avoid token limits
    const truncated = truncateContext(messages);
    const fullMessages = [{ role: 'system', content: SYSTEM_PROMPT }, ...truncated];

    const useModel = model || 'groq';
    let response, usedModel = '';

    // Model fallback chain: primary -> fallback -> lighter -> together
    const MODELS = {
      primary: 'openai/gpt-oss-120b',
      fallback: 'openai/gpt-oss-20b',
      light: 'qwen/qwen3.6-27b'
    };

    // Try Groq with automatic fallback
    if (useModel === 'groq' && groq) {
      const tryModels = [MODELS.primary, MODELS.fallback, MODELS.light];
      for (const tryModel of tryModels) {
        try {
          usedModel = tryModel;
          response = await groq.chat.completions.create({
            model: usedModel,
            messages: fullMessages,
            tools: TOOLS,
            tool_choice: 'auto',
            temperature: 0.7,
            max_tokens: 4096
          });
          break; // success
        } catch (err) {
          if (err.status === 429 && tryModel !== MODELS.light) {
            continue; // try next model
          }
          throw err; // other error
        }
      }
      // If Groq failed completely, try Together AI
      if (!response && together) {
        try {
          usedModel = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo';
          response = await together.chat.completions.create({
            model: usedModel,
            messages: fullMessages,
            temperature: 0.7,
            max_tokens: 4096
          });
        } catch (err) {
          // Together also failed
        }
      }
      // If Together failed, try HuggingFace
      if (!response && hf) {
        try {
          usedModel = 'meta-llama/Meta-Llama-3.1-70B-Instruct';
          response = await hf.chat.completions.create({
            model: usedModel,
            messages: fullMessages,
            temperature: 0.7,
            max_tokens: 4096
          });
        } catch (err) {
          // HF also failed
        }
      }
    }
    // Try OpenAI
    else if (useModel === 'openai' && openai) {
      usedModel = 'gpt-4o';
      response = await openai.chat.completions.create({
        model: usedModel,
        messages: fullMessages,
        tools: TOOLS,
        tool_choice: 'auto',
        temperature: 0.7,
        max_tokens: 4096
      });
    }
    else if (apiKey) {
      const userOpenAI = new OpenAI({ apiKey });
      usedModel = 'gpt-4o';
      response = await userOpenAI.chat.completions.create({
        model: usedModel,
        messages: fullMessages,
        tools: TOOLS,
        tool_choice: 'auto',
        temperature: 0.7,
        max_tokens: 4096
      });
    }
    // Try Together AI as standalone fallback
    else if (together) {
      usedModel = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo';
      response = await together.chat.completions.create({
        model: usedModel,
        messages: fullMessages,
        temperature: 0.7,
        max_tokens: 4096
      });
    }
    // Try HuggingFace as standalone fallback
    else if (hf) {
      usedModel = 'meta-llama/Meta-Llama-3.1-70B-Instruct';
      response = await hf.chat.completions.create({
        model: usedModel,
        messages: fullMessages,
        temperature: 0.7,
        max_tokens: 4096
      });
    }
    else {
      return res.status(500).json({ error: 'Aucun provider AI configure.' });
    }

    const choice = response.choices[0];
    const assistantMessage = choice.message;

    // Handle Function Calling
    if (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
      const toolResults = [];

      for (const tc of assistantMessage.tool_calls) {
        let fnArgs;
        try { fnArgs = JSON.parse(tc.function.arguments); } catch (e) { fnArgs = {}; }

        let result;
        switch (tc.function.name) {
          case 'create_excel':
            result = { action: 'create_excel', filename: fnArgs.filename, data: fnArgs.data };
            break;
          case 'create_pdf':
            result = { action: 'create_pdf', filename: fnArgs.filename, title: fnArgs.title, paragraphs: fnArgs.paragraphs, markdown: fnArgs.markdown };
            break;
          case 'web_search':
            result = await webSearch(fnArgs.query, fnArgs.numResults);
            break;
          case 'run_code':
            result = await runCode(fnArgs.code, fnArgs.language);
            break;
          default:
            result = { error: 'Outil inconnu: ' + tc.function.name };
        }

        toolResults.push({
          role: 'tool',
          tool_call_id: tc.id,
          content: JSON.stringify(result)
        });
      }

      // Send tool results back for final response
      const followUp = [...fullMessages, assistantMessage, ...toolResults];
      let followUpResponse;

      if (useModel === 'groq' && groq) {
        followUpResponse = await groq.chat.completions.create({
          model: 'llama-3.1-70b-versatile',
          messages: followUp,
          temperature: 0.7,
          max_tokens: 4096
        });
      } else if (useModel === 'openai' && openai) {
        followUpResponse = await openai.chat.completions.create({
          model: 'gpt-4o',
          messages: followUp,
          temperature: 0.7,
          max_tokens: 4096
        });
      } else if (apiKey) {
        const userOpenAI = new OpenAI({ apiKey });
        followUpResponse = await userOpenAI.chat.completions.create({
          model: 'gpt-4o',
          messages: followUp,
          temperature: 0.7,
          max_tokens: 4096
        });
      }

      return res.json({
        content: followUpResponse.choices[0].message.content,
        model: usedModel,
        tool_calls: assistantMessage.tool_calls.map(tc => ({
          name: tc.function.name,
          args: JSON.parse(tc.function.arguments)
        })),
        usage: response.usage
      });
    }

    // No tool calls
    return res.json({
      content: assistantMessage.content,
      model: usedModel,
      usage: response.usage
    });

  } catch (error) {
    console.error('Chat error:', error.message);
    return res.status(500).json({ error: error.message });
  }
});

// ═══════════════════════════════════════════════════════════
// STATUS
// ═══════════════════════════════════════════════════════════
app.get('/api/status', (req, res) => {
  res.json({
    status: 'ok',
    providers: { openai: !!openai, groq: !!groq },
    version: 'V12 AI v2.1'
  });
});

app.get('/health', (req, res) => { res.json({ ok: true }); });

app.listen(PORT, () => {
  console.log(`V12 AI Proxy v2.1 running on port ${PORT}`);
  console.log(`OpenAI: ${openai ? 'configured' : 'not configured'}`);
  console.log(`Groq: ${groq ? 'configured' : 'not configured'}`);
});
