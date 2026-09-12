require('dotenv').config();
const express = require('express');
const cors = require('cors');
const OpenAI = require('openai');
const Groq = require('groq-sdk');
const axios = require('axios');

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
// IMAGE ANALYSIS with Groq Vision (llama-4-scout)
// ═══════════════════════════════════════════════════════════
async function analyzeWithVision(fullMessages, image) {
  if (!groq) { console.log('[VISION] No groq'); return null; }
  
  // Ensure only the last user message has image array, rest are strings
  const visionMessages = fullMessages.map(m => ({
    role: m.role,
    content: Array.isArray(m.content) ? m.content.map(c => c.text || '').join(' ') : m.content
  }));
  
  // Add image to last user message
  const lastUser = visionMessages.findLast(m => m.role === 'user');
  if (lastUser) {
    lastUser.content = [
      { type: 'text', text: 'Decris en detail ce que tu vois dans cette image. Identifie les objets, les personnes, le texte, les couleurs, et tout autre detail pertinent.' },
      { type: 'image_url', image_url: { url: image } }
    ];
  }
  
  console.log('[VISION] Messages count:', visionMessages.length);
  console.log('[VISION] Last msg role:', lastUser?.role, 'content type:', Array.isArray(lastUser?.content) ? 'array' : 'string');
  
  const visionModels = ['meta-llama/llama-4-scout-17b-16e-instruct'];
  for (const vm of visionModels) {
    try {
      console.log('[VISION] Trying', vm);
      const res = await groq.chat.completions.create({
        model: vm,
        messages: visionMessages,
        temperature: 0.7,
        max_tokens: 1500
      });
      console.log('[VISION] Success with', vm);
      return res.choices[0].message.content;
    } catch (err) {
      console.log('[VISION] Failed:', err.status, err.message?.substring(0, 100));
      continue;
    }
  }
  return null;
}

// ═══════════════════════════════════════════════════════════
// IMAGE ANALYSIS with HuggingFace (BLIP captioning)
// ═══════════════════════════════════════════════════════════
async function analyzeWithHF(base64Image) {
  if (!HF_KEY) { console.log('[HF] No key'); return null; }
  try {
    const base64Data = base64Image.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(base64Data, 'base64');
    console.log('[HF] Sending image, buffer size:', imageBuffer.length);
    
    const captionRes = await axios.post(
      'https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base',
      imageBuffer,
      { headers: { 'Authorization': `Bearer ${HF_KEY}`, 'Content-Type': 'application/octet-stream' }, timeout: 120000 }
    );
    console.log('[HF] Response:', JSON.stringify(captionRes.data).substring(0, 200));
    const caption = captionRes.data[0]?.generated_text || captionRes.data.generated_text || '';
    return caption;
  } catch (err) {
    console.error('[HF] Error:', err.response?.data ? JSON.stringify(err.response.data).substring(0, 200) : err.message);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════
// SYSTEM PROMPT V12 AI — Structure Role/Consignes/Format
// ═══════════════════════════════════════════════════════════
const SYSTEM_PROMPT = `Tu es V12 AI, developpe par l'equipe V12. Reponds en francais, sois concis.

IMPORTANT: Utilise les outils SEULEMENT quand l'utilisateur le demande explicitement.
- Salutations, questions simples, conversation: reponds directement SANS outil.
- "joue", "ecoute": utilise play_music
- "dessine", "image", "genere": utilise generate_image
- "cree un CV": utilise create_cv
- "cree un PDF": utilise create_pdf
- "cree un Excel": utilise create_excel
- "recherche": utilise web_search
- "execute du code": utilise run_code

Tu connais la RDC et l'Afrique. NE DIS PAS que tu es OpenAI.`;

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
  },
  {
    type: 'function',
    function: {
      name: 'create_cv',
      description: 'Cree un CV professionnel au format PDF. Utilise quand l\'utilisateur veut un curriculum vitae, un resume professionnel, ou un profil pour candidature.',
      parameters: {
        type: 'object',
        properties: {
          filename: { type: 'string', description: 'Nom du fichier sans extension (ex: CV_Jean_Dupont)' },
          firstName: { type: 'string', description: 'Prenom' },
          lastName: { type: 'string', description: 'Nom de famille' },
          title: { type: 'string', description: 'Titre professionnel ou poste vise (ex: Developpeur Full Stack)' },
          email: { type: 'string', description: 'Adresse email' },
          phone: { type: 'string', description: 'Numero de telephone' },
          city: { type: 'string', description: 'Ville de residence' },
          country: { type: 'string', description: 'Pays de residence' },
          summary: { type: 'string', description: 'Resume professionnel (2-3 lignes)' },
          experiences: {
            type: 'array',
            description: 'Liste des experiences professionnelles',
            items: {
              type: 'object',
              properties: {
                period: { type: 'string', description: 'Periode (ex: 2020-2023)' },
                role: { type: 'string', description: 'Intitule du poste' },
                company: { type: 'string', description: 'Nom de l\'entreprise' },
                location: { type: 'string', description: 'Lieu' },
                description: { type: 'string', description: 'Description des missions et realisations' }
              }
            }
          },
          education: {
            type: 'array',
            description: 'Liste des formations',
            items: {
              type: 'object',
              properties: {
                year: { type: 'string', description: 'Annee d\'obtention' },
                degree: { type: 'string', description: 'Diplome ou titre' },
                school: { type: 'string', description: 'Etablissement' },
                details: { type: 'string', description: 'Details supplementaires' }
              }
            }
          },
          skills: {
            type: 'array',
            description: 'Liste des competences techniques',
            items: { type: 'string' }
          },
          languages: {
            type: 'array',
            description: 'Liste des langues avec niveau',
            items: { type: 'string' }
          },
          interests: {
            type: 'array',
            description: 'Centres d\'interet',
            items: { type: 'string' }
          }
        },
        required: ['filename', 'firstName', 'lastName', 'title']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'play_music',
      description: 'Joue de la musique. Utilise quand l\'utilisateur veut ecouter de la musique, jouer une chanson, ou lancer un morceau. Utilise YouTube comme source.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Recherche musicale (ex: "Fally Ipupa Tokooos", "Ed Sheeran Shape of You", "musique africaine relaxante")' },
          action: { type: 'string', enum: ['play', 'stop'], description: 'Action: play pour lancer, stop pour arreter' }
        },
        required: ['query']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'generate_image',
      description: 'Genere une image a partir d\'une description textuelle. Utilise quand l\'utilisateur veut un dessin, une image, un design, ou genere une image.',
      parameters: {
        type: 'object',
        properties: {
          prompt: { type: 'string', description: 'Description detaillee de l\'image en anglais (ex: "a beautiful African woman wearing traditional clothing, oil painting style, warm colors")' },
          style: { type: 'string', description: 'Style: photo, anime, painting, 3d, digital_art, etc.' },
          filename: { type: 'string', description: 'Nom du fichier (optionnel)' }
        },
        required: ['prompt']
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
    const { messages, model, apiKey, image } = req.body;
    if (!messages || !Array.isArray(messages)) return res.status(400).json({ error: 'messages requis' });

    // Truncate context to avoid token limits
    const truncated = truncateContext(messages);
    const fullMessages = [{ role: 'system', content: SYSTEM_PROMPT }, ...truncated];

    // Ensure all messages have string content (fix array content from previous vision calls)
    fullMessages.forEach(m => {
      if (Array.isArray(m.content)) {
        m.content = m.content.map(c => c.text || '').join(' ');
      }
    });

    const useModel = model || 'groq';
    let response, usedModel = '';

    // If image provided, try vision model directly
    if (image) {
      console.log('[IMAGE] Received image, length:', image.length);
      try {
        // First try Groq vision model (direct image understanding)
        const visionReply = await analyzeWithVision(fullMessages, image);
        if (visionReply) {
          console.log('[IMAGE] Vision model succeeded');
          const lastUserMsg = fullMessages.findLast(m => m.role === 'user');
          if (lastUserMsg) {
            lastUserMsg.content = lastUserMsg.content.replace(/\n\[Image jointe\]/, '') + '\n\n[Contenu de l\'image analysee]\n' + visionReply;
          }
        } else {
          // Fallback to HuggingFace captioning
          console.log('[IMAGE] Vision failed, trying HF...');
          const hfCaption = await analyzeWithHF(image);
          const lastUserMsg = fullMessages.findLast(m => m.role === 'user');
          if (lastUserMsg) {
            const baseText = lastUserMsg.content.replace(/\n\[Image jointe\]/, '');
            if (hfCaption) {
              lastUserMsg.content = baseText + '\n\n[Description de l\'image]\n' + hfCaption;
            } else {
              lastUserMsg.content = baseText + '\n\n[L\'utilisateur a envoye une image. Les outils d\'analyse ne sont pas disponibles. Demandez-lui de decrire l\'image.]';
            }
          }
        }
      } catch (imgErr) {
        console.error('[IMAGE] Error:', imgErr.message);
        const lastUserMsg = fullMessages.findLast(m => m.role === 'user');
        if (lastUserMsg) {
          lastUserMsg.content = lastUserMsg.content.replace(/\n\[Image jointe\]/, '') + '\n\n[L\'utilisateur a envoye une image. Demandez-lui de la decrire.]';
        }
      }
    }

    // Model fallback chain: primary -> fallback -> lighter -> together
    const MODELS = {
      primary: 'openai/gpt-oss-120b',
      fallback: 'openai/gpt-oss-20b',
      light: 'qwen/qwen3.6-27b'
    };

      // Try Groq with automatic fallback (text-only)
    if (!response && useModel === 'groq' && groq) {
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
            max_tokens: 1500
          });
          break; // success
        } catch (err) {
          console.error('Groq model ' + tryModel + ' failed:', err.status, err.message);
          if (err.status === 429) {
            continue; // try next model
          }
          continue; // also continue on other errors
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
            max_tokens: 1500
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
            max_tokens: 1500
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
        max_tokens: 1500
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
        max_tokens: 1500
      });
    }
    // Try Together AI as standalone fallback
    else if (together) {
      usedModel = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo';
      response = await together.chat.completions.create({
        model: usedModel,
        messages: fullMessages,
        temperature: 0.7,
        max_tokens: 1500
      });
    }
    // Try HuggingFace as standalone fallback
    else if (hf) {
      usedModel = 'meta-llama/Meta-Llama-3.1-70B-Instruct';
      response = await hf.chat.completions.create({
        model: usedModel,
        messages: fullMessages,
        temperature: 0.7,
        max_tokens: 1500
      });
    }
    else {
      return res.status(500).json({ error: 'Aucun provider AI configure.' });
    }

    if (!response || !response.choices || !response.choices[0]) {
      return res.status(503).json({ error: { message: 'Le serveur demarre. Attendez 30 secondes puis reessayez.' } });
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
          case 'create_cv':
            result = { action: 'create_cv', ...fnArgs };
            break;
          case 'play_music':
            result = { action: 'play_music', query: fnArgs.query, action: fnArgs.action || 'play' };
            break;
          case 'generate_image':
            result = { action: 'generate_image', prompt: fnArgs.prompt, style: fnArgs.style, filename: fnArgs.filename };
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
        const followUpModels = ['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3.6-27b'];
        for (const fuModel of followUpModels) {
          try {
            followUpResponse = await groq.chat.completions.create({
              model: fuModel,
              messages: followUp,
              temperature: 0.7,
              max_tokens: 1500
            });
            break;
          } catch (e) { continue; }
        }
      } else if (useModel === 'openai' && openai) {
        followUpResponse = await openai.chat.completions.create({
          model: 'gpt-4o',
          messages: followUp,
          temperature: 0.7,
          max_tokens: 1500
        });
      } else if (apiKey) {
        const userOpenAI = new OpenAI({ apiKey });
        followUpResponse = await userOpenAI.chat.completions.create({
          model: 'gpt-4o',
          messages: followUp,
          temperature: 0.7,
          max_tokens: 1500
        });
      }

      // If follow-up failed, return tool results directly
      if (!followUpResponse || !followUpResponse.choices || !followUpResponse.choices[0]) {
        var toolSummary = toolResults.map(function(tr) {
          var r = JSON.parse(tr.content);
          return r.error || 'Action executee';
        }).join('\n');
        return res.json({
          content: toolSummary || 'Action executee avec succes.',
          model: usedModel,
          tool_calls: assistantMessage.tool_calls.map(tc => ({
            function: tc.function.name,
            params: JSON.parse(tc.function.arguments)
          })),
          usage: response.usage
        });
      }

      return res.json({
        content: followUpResponse.choices[0].message.content,
        model: usedModel,
        tool_calls: assistantMessage.tool_calls.map(tc => ({
          function: tc.function.name,
          params: JSON.parse(tc.function.arguments)
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
