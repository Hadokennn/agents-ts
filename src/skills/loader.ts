import fs from 'node:fs';
import path from 'node:path';

export interface SkillDefinition {
  name: string;
  description: string;
  whenToUse?: string;
  content: string;
  dirPath: string;
}

const SKILLS_DIR = '.skills';
const SKILL_FILE = 'SKILL.md';

export class SkillLoader {
  private readonly baseDir: string;
  private skills = new Map<string, SkillDefinition>();

  constructor(baseDir = '.') {
    this.baseDir = baseDir;
  }

  load(): SkillDefinition[] {
    this.skills.clear();
    const skillsDir = path.join(this.baseDir, SKILLS_DIR);
    if (!fs.existsSync(skillsDir)) return [];

    const entries = fs.readdirSync(skillsDir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const skillFile = path.join(skillsDir, entry.name, SKILL_FILE);
      if (!fs.existsSync(skillFile)) continue;

      const raw = fs.readFileSync(skillFile, 'utf-8');
      const parsed = this.parseFrontmatter(raw);
      if (!parsed) continue;

      this.skills.set(entry.name, {
        name: entry.name,
        description: parsed.description,
        content: parsed.content,
        dirPath: path.join(skillsDir, entry.name),
      });
    }
    return this.list();
  }

  list(): SkillDefinition[] {
    return Array.from(this.skills.values());
  }

  get(name: string): SkillDefinition | undefined {
    return this.skills.get(name);
  }

  // 这里没有LLM自动加载SKILL，只做的半自动的，用户输入 /skill load code-review 后，SkillLoader 把 code-review 的完整内容塞进系统提示词
  buildPromptSection(activeSkills: Set<string>): string | null {
    if (this.skills.size === 0) return null;
    const lines: string[] = [];

    for (const name of activeSkills) {
      const skill = this.skills.get(name);
      if (!skill) continue;
      lines.push(`[激活的 Skill: ${skill.name}]`);
      lines.push(skill.content);
      lines.push('');
    }

    const available = this.list()
      .filter(s => !activeSkills.has(s.name))
      .map(s => `  /${s.name} — ${s.description}`);
    if (available.length > 0) {
      lines.push('可用的 Skills（输入 /skill load <name> 激活）：');
      lines.push(...available);
    }
    return lines.length > 0 ? lines.join('\n') : null;
  }

  private parseFrontmatter(raw: string): { description: string; content: string } | null {
    const match = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
    if (!match) return { description: '', content: raw };
    const meta: Record<string, string> = {};
    for (const line of match[1].split('\n')) {
      const idx = line.indexOf(':');
      if (idx > 0) {
        const key = line.slice(0, idx).trim();
        let value = line.slice(idx + 1).trim();
        if (value.startsWith('"') && value.endsWith('"')) value = value.slice(1, -1);
        meta[key] = value;
      }
    }
    return { description: meta.description || '', content: match[2].trim() };
  }
}
