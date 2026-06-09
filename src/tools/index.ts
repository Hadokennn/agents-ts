import type { ToolDefinition } from './registry.js';
import { weatherTool, calculatorTool } from './utility-tools.js';
import { readFileTool, writeFileTool, editFileTool, listDirectoryTool, startPreviewTool } from './file-tools.js';
import { globTool, grepTool } from './search-tools.js';
import { bashTool } from './shell-tools.js';
import { pickSearchTool, webFetchTool } from './web-search.js';

export const allTools: ToolDefinition[] = [
  weatherTool,
  calculatorTool,
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
  weatherTool, calculatorTool,
  readFileTool, writeFileTool, editFileTool, listDirectoryTool,
  globTool, grepTool,
  bashTool,
};
