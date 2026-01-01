import React from 'react';
import './ChatWindow.css';

interface ChatIconProps {
  onClick: () => void;
  isOpen: boolean;
}

export const ChatIcon: React.FC<ChatIconProps> = ({ onClick, isOpen }) => {
  return (
    <button
      className={`chat-icon ${isOpen ? 'open' : ''}`}
      onClick={onClick}
      aria-label="Open chat"
    >
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
      </svg>
    </button>
  );
};
