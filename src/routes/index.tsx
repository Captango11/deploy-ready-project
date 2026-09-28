import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "CareerMatch AI — NLP Resume Intelligence" },
      { name: "description", content: "NLP + ML resume classifier built with Streamlit, ready to deploy on Render." },
      { property: "og:title", content: "CareerMatch AI — NLP Resume Intelligence" },
      { property: "og:description", content: "NLP + ML resume classifier built with Streamlit, ready to deploy on Render." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <main className="min-h-screen bg-background text-foreground p-10 max-w-3xl mx-auto space-y-4">
      <h1 className="text-4xl font-bold">CareerMatch AI</h1>
      <p className="text-muted-foreground">NLP-Powered Resume Intelligence &amp; Career Matching</p>
      <p>
        This is a Python Streamlit app located in the <code>career-match-ai/</code> folder. It runs locally or on
        Render — it cannot run inside this preview.
      </p>
      <ol className="list-decimal pl-6 space-y-1">
        <li>Connect this project to GitHub and push.</li>
        <li>On Render choose New → Blueprint and pick the repo (render.yaml is included).</li>
        <li>Open the Render URL once the build finishes.</li>
      </ol>
    </main>
  );
}
