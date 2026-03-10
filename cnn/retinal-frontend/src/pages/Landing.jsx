import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Eye,
  Brain,
  ShieldCheck,
  ArrowRight,
  Zap,
  Activity,
  BarChart3,
  Upload,
  Cpu,
  FileText,
  CheckCircle2,
  AlertTriangle,
  ChevronRight,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { PageTransition } from '@/components/layout/PageTransition';

const stats = [
  { value: '45', label: 'Disease Classes', icon: Eye },
  { value: '0.82', label: 'Best AUC Score', icon: Brain },
  { value: '<1s', label: 'Inference Time', icon: Zap },
  { value: 'FREE', label: 'Open Access', icon: ShieldCheck },
];

const features = [
  {
    icon: Brain,
    title: 'Multi-Label CNN',
    description:
      'A deep convolutional network trained to detect 45 distinct retinal conditions simultaneously from a single fundus image.',
    badge: 'AI Model',
    badgeVariant: 'default',
  },
  {
    icon: Zap,
    title: 'Sub-Second Results',
    description:
      'Optimised GPU inference pipeline delivers predictions in under a second — fast enough for high-throughput screening.',
    badge: 'Performance',
    badgeVariant: 'secondary',
  },
  {
    icon: BarChart3,
    title: 'Confidence Scores',
    description:
      'Every disease class is returned with a calibrated probability score, sortable and filterable by threshold.',
    badge: 'Explainable',
    badgeVariant: 'secondary',
  },
  {
    icon: ShieldCheck,
    title: 'Clinical Advisory',
    description:
      'Detected conditions trigger automated advisory notes and risk-level flags (HIGH / MODERATE / LOW) for clinician triage.',
    badge: 'Clinical',
    badgeVariant: 'secondary',
  },
  {
    icon: FileText,
    title: 'Detailed Reports',
    description:
      'Full results include disease short codes, human-readable full names, confidence percentages, and a medical disclaimer.',
    badge: 'Reporting',
    badgeVariant: 'secondary',
  },
  {
    icon: Cpu,
    title: 'REST API',
    description:
      'Backend exposed as a FastAPI service — easy to integrate with EMR systems, PACS pipelines, or mobile apps via JSON.',
    badge: 'API',
    badgeVariant: 'secondary',
  },
];

const pipeline = [
  { step: '01', icon: Upload, title: 'Upload Image', desc: 'Upload any eye image (fundus, OCT, close-up) up to 20 MB.' },
  { step: '02', icon: Cpu,    title: 'AI Inference',  desc: 'CNN processes the image at 384 × 384 with ImageNet normalisation.' },
  { step: '03', icon: FileText, title: 'Get Results', desc: 'Receive 45-class probabilities, risk level, and a clinical advisory.' },
];

const sampleDiseases = [
  { code: 'DR',   name: 'Diabetic Retinopathy',           risk: true  },
  { code: 'ARMD', name: 'Age-Related Macular Degeneration', risk: true  },
  { code: 'BRVO', name: 'Branch Retinal Vein Occlusion',  risk: true  },
  { code: 'MH',   name: 'Macular Hole',                   risk: false },
  { code: 'ERM',  name: 'Epiretinal Membrane',            risk: false },
  { code: 'CSR',  name: 'Central Serous Retinopathy',     risk: false },
  { code: 'CRVO', name: 'Central Retinal Vein Occlusion', risk: true  },
  { code: 'VH',   name: 'Vitreous Hemorrhage',            risk: true  },
  { code: 'RP',   name: 'Retinitis Pigmentosa',           risk: true  },
  { code: 'CME',  name: 'Cystoid Macular Edema',          risk: true  },
  { code: 'ODC',  name: 'Optic Disc Cupping',             risk: false },
  { code: 'MYA',  name: 'Myopic Astigmatism',             risk: false },
];

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 20 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true },
  transition: { duration: 0.45, delay },
});

const stagger = {
  container: { animate: { transition: { staggerChildren: 0.08 } } },
  item: { initial: { opacity: 0, y: 18 }, animate: { opacity: 1, y: 0, transition: { duration: 0.4 } } },
};

export default function Landing() {
  return (
    <PageTransition>
      <div className="relative overflow-hidden">

        {/* ── Hero ─────────────────────────────────────────────────────── */}
        <section className="relative mx-auto max-w-7xl px-4 sm:px-6 pt-20 pb-28">
          <div className="pointer-events-none absolute -top-40 -right-40 h-[560px] w-[560px] rounded-full bg-primary/5 blur-3xl" />
          <div className="pointer-events-none absolute bottom-0 -left-40 h-[400px] w-[400px] rounded-full bg-primary/5 blur-3xl" />

          <div className="relative text-center space-y-8 max-w-3xl mx-auto">
            <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
              <span className="inline-flex items-center gap-2 rounded-full bg-primary/10 border border-primary/20 px-4 py-1.5 text-sm font-medium text-primary">
                <Activity className="h-3.5 w-3.5" />
                AI-Powered Medical Imaging
              </span>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.6 }}
              className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight leading-[1.1]"
            >
              Retinal Disease{' '}
              <span className="text-primary">AI Classifier</span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.5 }}
              className="text-lg text-muted-foreground leading-relaxed max-w-2xl mx-auto"
            >
              Upload any eye image and get instant AI-powered disease
              predictions across 45 conditions — with confidence scores, risk levels,
              and clinical advisory notes.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.4 }}
              className="flex flex-wrap justify-center gap-3"
            >
              <Link to="/predict">
                <Button size="lg" className="text-base shadow-lg shadow-primary/20">
                  <Eye className="h-5 w-5" />
                  Start Analysis
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link to="/how-it-works">
                <Button size="lg" variant="outline" className="text-base">
                  How It Works
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </Link>
            </motion.div>

            {/* Trust line */}
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="text-xs text-muted-foreground/60"
            >
              For screening purposes only · Not a medical diagnostic device
            </motion.p>
          </div>
        </section>

        {/* ── Stats strip ──────────────────────────────────────────────── */}
        <section className="border-y border-border/50 bg-muted/30">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 py-10">
            <motion.div
              variants={stagger.container}
              initial="initial"
              whileInView="animate"
              viewport={{ once: true, amount: 0.3 }}
              className="grid grid-cols-2 md:grid-cols-4 gap-6"
            >
              {stats.map(({ value, label, icon: Icon }) => (
                <motion.div key={label} variants={stagger.item} className="text-center space-y-2">
                  <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10">
                    <Icon className="h-5 w-5 text-primary" />
                  </div>
                  <p className="text-2xl font-black text-foreground">{value}</p>
                  <p className="text-xs text-muted-foreground">{label}</p>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </section>

        {/* ── Features grid ────────────────────────────────────────────── */}
        <section className="mx-auto max-w-7xl px-4 sm:px-6 py-20">
          <motion.div {...fadeUp()} className="text-center mb-12">
            <h2 className="text-3xl font-black text-foreground">What It Does</h2>
            <p className="text-muted-foreground mt-2 max-w-xl mx-auto">
              A full end-to-end retinal screening pipeline — from raw image to structured clinical output.
            </p>
          </motion.div>

          <motion.div
            variants={stagger.container}
            initial="initial"
            whileInView="animate"
            viewport={{ once: true, amount: 0.1 }}
            className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5"
          >
            {features.map(({ icon: Icon, title, description, badge, badgeVariant }) => (
              <motion.div key={title} variants={stagger.item}>
                <Card className="h-full hover:shadow-lg hover:border-primary/25 transition-all duration-300 group">
                  <CardContent className="p-6 space-y-4">
                    <div className="flex items-start justify-between">
                      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 group-hover:bg-primary/15 transition-colors">
                        <Icon className="h-6 w-6 text-primary" />
                      </div>
                      <Badge variant={badgeVariant} className="text-xs">{badge}</Badge>
                    </div>
                    <h3 className="font-bold text-foreground">{title}</h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">{description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </section>

        {/* ── Pipeline teaser ──────────────────────────────────────────── */}
        <section className="border-y border-border/50 bg-muted/20">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 py-16">
            <motion.div {...fadeUp()} className="text-center mb-10">
              <h2 className="text-3xl font-black text-foreground">Three Steps to Results</h2>
              <p className="text-muted-foreground mt-2">
                The entire pipeline from upload to diagnosis in seconds.
              </p>
            </motion.div>

            <div className="relative grid sm:grid-cols-3 gap-6">
              {/* connector line (desktop) */}
              <div className="hidden sm:block absolute top-10 left-[calc(16.67%+1.5rem)] right-[calc(16.67%+1.5rem)] h-px bg-border/60" />

              {pipeline.map(({ step, icon: Icon, title, desc }, i) => (
                <motion.div key={step} {...fadeUp(i * 0.12)} className="relative text-center space-y-4">
                  <div className="mx-auto flex h-20 w-20 flex-col items-center justify-center rounded-2xl bg-background border border-border shadow-sm z-10 relative">
                    <Icon className="h-8 w-8 text-primary" />
                    <span className="text-[10px] font-bold text-muted-foreground mt-1">{step}</span>
                  </div>
                  <div>
                    <p className="font-bold text-foreground">{title}</p>
                    <p className="text-sm text-muted-foreground mt-1 max-w-[200px] mx-auto">{desc}</p>
                  </div>
                </motion.div>
              ))}
            </div>

            <motion.div {...fadeUp(0.3)} className="text-center mt-10">
              <Link to="/how-it-works">
                <Button variant="outline">
                  Full Pipeline Walkthrough
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </motion.div>
          </div>
        </section>

        {/* ── Disease coverage ─────────────────────────────────────────── */}
        <section className="mx-auto max-w-7xl px-4 sm:px-6 py-20">
          <motion.div {...fadeUp()} className="text-center mb-10">
            <h2 className="text-3xl font-black text-foreground">45 Conditions Screened</h2>
            <p className="text-muted-foreground mt-2 max-w-xl mx-auto">
              The model simultaneously evaluates all of the following retinal conditions in a
              single inference pass.
            </p>
          </motion.div>

          <motion.div
            variants={stagger.container}
            initial="initial"
            whileInView="animate"
            viewport={{ once: true, amount: 0.1 }}
            className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3"
          >
            {sampleDiseases.map(({ code, name, risk }) => (
              <motion.div key={code} variants={stagger.item}>
                <Card className="hover:shadow-md transition-shadow">
                  <CardContent className="p-4 flex items-center gap-3">
                    {risk
                      ? <AlertTriangle className="h-4 w-4 text-red-500 shrink-0" />
                      : <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
                    }
                    <div className="min-w-0">
                      <p className="text-xs font-bold text-foreground">{code}</p>
                      <p className="text-xs text-muted-foreground truncate">{name}</p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>

          <motion.div {...fadeUp(0.2)} className="text-center mt-8">
            <p className="text-sm text-muted-foreground">
              <span className="inline-flex items-center gap-1.5 mr-3">
                <AlertTriangle className="h-3.5 w-3.5 text-red-500" /> High-risk condition
              </span>
              <span className="inline-flex items-center gap-1.5">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" /> Standard finding
              </span>
              <span className="ml-3 text-muted-foreground/60">· and 33 more conditions</span>
            </p>
          </motion.div>
        </section>

        {/* ── CTA ──────────────────────────────────────────────────────── */}
        <section className="mx-auto max-w-7xl px-4 sm:px-6 pb-24">
          <motion.div {...fadeUp()}>
            <Card className="bg-gradient-to-br from-primary/15 via-primary/5 to-transparent border-primary/20 overflow-hidden relative">
              <div className="pointer-events-none absolute -top-20 -right-20 h-64 w-64 rounded-full bg-primary/10 blur-3xl" />
              <CardContent className="relative p-10 sm:p-14 flex flex-col sm:flex-row items-center justify-between gap-8">
                <div className="space-y-3 text-center sm:text-left">
                  <h3 className="text-3xl font-black text-foreground">
                    Ready to screen?
                  </h3>
                  <p className="text-muted-foreground max-w-sm">
                    Upload any eye image now and receive a full AI-powered retinal
                    disease screening report in seconds.
                  </p>
                  <p className="text-xs text-muted-foreground/50">
                    Free to use · No account required · Results in &lt;1 second
                  </p>
                </div>
                <div className="flex flex-col gap-3 shrink-0">
                  <Link to="/predict">
                    <Button size="lg" className="text-base w-full shadow-lg shadow-primary/20">
                      <Eye className="h-5 w-5" />
                      Start Analysis
                      <ArrowRight className="h-4 w-4" />
                    </Button>
                  </Link>
                  <Link to="/how-it-works">
                    <Button size="lg" variant="outline" className="text-base w-full">
                      Learn How It Works
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </section>

      </div>
    </PageTransition>
  );
}
