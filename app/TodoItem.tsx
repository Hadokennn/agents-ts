import React, { useState } from 'react';

interface TodoItemProps {
  id: string;
  text: string;
  completed: boolean;
  onToggle: (id: string) => void;
  onDelete: (id: string) => void;
  onEdit: (id: string, text: string) => void;
}

export function TodoItem({ id, text, completed, onToggle, onDelete, onEdit }: TodoItemProps) {
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState(text);

  const handleSave = () => {
    const trimmed = editText.trim();
    if (trimmed) {
      onEdit(id, trimmed);
    }
    setEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave();
    } else if (e.key === 'Escape') {
      setEditText(text);
      setEditing(false);
    }
  };

  return (
    <li className={`todo-item ${completed ? 'completed' : ''}`}>
      <input
        type="checkbox"
        className="todo-checkbox"
        checked={completed}
        onChange={() => onToggle(id)}
      />

      {editing ? (
        <input
          className="todo-text-input"
          value={editText}
          onChange={(e) => setEditText(e.target.value)}
          onBlur={handleSave}
          onKeyDown={handleKeyDown}
          autoFocus
        />
      ) : (
        <span className="todo-text">{text}</span>
      )}

      <div className="todo-actions">
        <button
          className="edit-btn"
          onClick={() => {
            setEditing(true);
            setEditText(text);
          }}
          title="编辑"
        >
          ✏️
        </button>
        <button
          className="delete-btn"
          onClick={() => onDelete(id)}
          title="删除"
        >
          🗑️
        </button>
      </div>
    </li>
  );
}
