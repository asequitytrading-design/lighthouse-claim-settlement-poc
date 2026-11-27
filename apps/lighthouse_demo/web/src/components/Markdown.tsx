import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function Markdown({ text }: { text: string }) {
  return (
    <div className="prose-chat">
      <ReactMarkdown remarkPlugins={[remarkGfm]}
                     components={{ a: (props) => <a {...props} target="_blank" rel="noreferrer" className="text-accent underline" /> }}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
