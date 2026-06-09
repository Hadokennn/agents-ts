import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { TodoItem } from './TodoItem.tsx';

type Filter = 'all' | 'active' | 'completed';

interface Todo {
  id: string;
  text: string;
  completed: boolean;
}

function App() {
  const [todos, setTodos] = useState<Todo[]>(() => {
    const saved = localStorage.getItem('green-todos');
    if (saved) {
      try {
        return JSON.parse(saved) as Todo[];
      } catch {
        return [];
      }
    }
    return [
      { id: crypto.randomUUID(), text: '🌱 给植物浇水', completed: false },
      { id: crypto.randomUUID(), text: '🌿 去公园散步', completed: false },
      { id: crypto.randomUUID(), text: '🍃 做深呼吸练习', completed: true },
    ];
  });

  const [input, setInput] = useState('');
  const [filter, setFilter] = useState<Filter>('all');

  // 持久化到 localStorage
  useEffect(() => {
    localStorage.setItem('green-todos', JSON.stringify(todos));
  }, [todos]);

  const addTodo = () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    setTodos((prev) => [
      ...prev,
      { id: crypto.randomUUID(), text: trimmed, completed: false },
    ]);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') addTodo();
  };

  const toggleTodo = (id: string) => {
    setTodos((prev) =>
      prev.map((t) => (t.id === id ? { ...t, completed: !t.completed } : t))
    );
  };

  const deleteTodo = (id: string) => {
    setTodos((prev) => prev.filter((t) => t.id !== id));
  };

  const editTodo = (id: string, text: string) => {
    setTodos((prev) =>
      prev.map((t) => (t.id === id ? { ...t, text } : t))
    );
  };

  const clearCompleted = () => {
    setTodos((prev) => prev.filter((t) => !t.completed));
  };

  const filteredTodos = todos.filter((t) => {
    if (filter === 'active') return !t.completed;
    if (filter === 'completed') return t.completed;
    return true;
  });

  const activeCount = todos.filter((t) => !t.completed).length;
  const completedCount = todos.filter((t) => t.completed).length;

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>待办清单</h1>
        <p className="subtitle">让生活像森林一样清新有序</p>
      </header>

      <div className="stats-bar">
        <span>
          🌳 <span className="count">{activeCount}</span> 项待完成
          {completedCount > 0 && (
            <span style={{ marginLeft: 8, color: '#90b890' }}>
              已完成 <span className="count">{completedCount}</span>
            </span>
          )}
        </span>
        {completedCount > 0 && (
          <button className="clear-btn" onClick={clearCompleted}>
            清除已完成
          </button>
        )}
      </div>

      <div className="add-todo-form">
        <input
          type="text"
          placeholder="写下新任务... 🌿"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button onClick={addTodo}>添加</button>
      </div>

      <div className="filter-bar">
        {(['all', 'active', 'completed'] as Filter[]).map((f) => (
          <button
            key={f}
            className={`filter-btn ${filter === f ? 'active' : ''}`}
            onClick={() => setFilter(f)}
          >
            {f === 'all' ? '全部' : f === 'active' ? '进行中' : '已完成'}
          </button>
        ))}
      </div>

      {filteredTodos.length === 0 ? (
        <div className="todo-list empty">
          {filter === 'all'
            ? '还没有待办，添加一个吧！'
            : filter === 'active'
            ? '全部完成啦，真棒！🎉'
            : '还没有已完成的待办'}
        </div>
      ) : (
        <ul className="todo-list">
          {filteredTodos.map((todo) => (
            <TodoItem
              key={todo.id}
              id={todo.id}
              text={todo.text}
              completed={todo.completed}
              onToggle={toggleTodo}
              onDelete={deleteTodo}
              onEdit={editTodo}
            />
          ))}
        </ul>
      )}

      <footer className="app-footer">
        <span>🌿 与自然同步 · {new Date().toLocaleDateString('zh-CN', { weekday: 'long' })}</span>
      </footer>
    </div>
  );
}

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
