import "@reactor-team/ui/styles.css";
import "./globals.css";

export const metadata = {
  title: "Prompt Token Studio",
  description:
    "Slice a prompt into labeled layers and see how each one fills the model's context window. Built on the Reactor developer platform.",
  icons: {
    icon: "/tv-icon.svg",
    shortcut: "/tv-icon.svg",
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" type="image/svg+xml" href="/tv-icon.svg" sizes="any" />
        <link rel="shortcut icon" type="image/svg+xml" href="/tv-icon.svg" />
      </head>
      <body>{children}</body>
    </html>
  );
}
