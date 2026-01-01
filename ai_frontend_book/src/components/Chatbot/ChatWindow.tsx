import React, { useState, useEffect, useRef } from 'react';
import { ChatIcon } from './ChatIcon';
import './ChatWindow.css';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
}

export const ChatWindow: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const welcomeMessage: Message = {
    id: 'welcome',
    text: 'Ask anything about this book. Answers come only from the book content.',
    sender: 'assistant',
  };

  // Add welcome message on mount
  useEffect(() => {
    setMessages([welcomeMessage]);
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = () => {
    if (inputValue.trim() === '') return;

    const newUserMessage: Message = {
      id: Date.now().toString(),
      text: inputValue.trim(),
      sender: 'user',
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setInputValue('');

    // TODO: Integrate with backend API here
    // For now, we'll simulate an assistant response
    setTimeout(() => {
      const assistantResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: 'I received your message. API integration pending.',
        sender: 'assistant',
      };
      setMessages((prev) => [...prev, assistantResponse]);
    }, 500);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <>
      <ChatIcon onClick={() => setIsOpen(!isOpen)} isOpen={isOpen} />

      <div className={`chat-window ${isOpen ? 'open' : ''}`}>
        <div className="chat-header">
          <h3>Book Assistant</h3>
          <button
            className="close-button"
            onClick={() => setIsOpen(false)}
            aria-label="Close chat"
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        <div className="chat-messages">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`message ${message.sender}`}
            >
              {message.text}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the book..."
            aria-label="Message input"
          />
          <button
            className="send-button"
            onClick={handleSendMessage}
            disabled={inputValue.trim() === ''}
            aria-label="Send message"
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </div>
    </>
  );
};
