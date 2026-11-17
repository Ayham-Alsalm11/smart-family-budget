/**
 * The agent's reply as simple Markdown (docs/ARCHITECTURE.md §10, decision D): bold, lists and lines only.
 * Raw HTML is dropped (`skipHtml`) and every other element, links included, is reduced to its text.
 */
import Markdown from "react-markdown";

const ALLOWED = ["p", "strong", "em", "ul", "ol", "li", "br"];

/** A single newline is a line break in the agent's text (the server's previews use them), not a space. */
function withLineBreaks(text: string): string {
  return text.replace(/\r\n/g, "\n").replace(/([^\n])\n(?!\n)/g, "$1  \n");
}

export default function MarkdownText({ text }: { text: string }) {
  return (
    <div className="space-y-2 leading-relaxed [&_ol]:list-decimal [&_ol]:ps-5 [&_ul]:list-disc [&_ul]:ps-5">
      <Markdown skipHtml allowedElements={ALLOWED} unwrapDisallowed>
        {withLineBreaks(text)}
      </Markdown>
    </div>
  );
}
