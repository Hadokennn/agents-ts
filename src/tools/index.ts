import type { ToolDefinition } from './registry.js';
import { readFileTool, writeFileTool, editFileTool, listDirectoryTool, startPreviewTool } from './file-tools.js';
import { globTool, grepTool } from './search-tools.js';
import { bashTool } from './shell-tools.js';
import { pickSearchTool, webFetchTool } from './web-search.js';

export const allTools: ToolDefinition[] = [
  readFileTool,
  writeFileTool,
  listDirectoryTool,
  editFileTool,
  globTool,
  grepTool,
  bashTool,
  startPreviewTool,
  pickSearchTool(),
  webFetchTool,
];

export {
  readFileTool, writeFileTool, editFileTool, listDirectoryTool,
  globTool, grepTool,
  bashTool,
  ToolDefinition,
};
