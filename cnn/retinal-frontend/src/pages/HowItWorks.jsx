import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  Upload,
  Cpu,
  BarChart3,
  ShieldCheck,
  FileText,
  ArrowRight,
  Brain,
  Eye,
  Zap,
  Activity,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { PageTransition } from '@/components/layout/PageTransition';

const steps = [
  {
    step: '01',
    icon: Upload,
    title: 'Upload an Eye Image',
    description:
      'Start by uploading any eye photograph — a retinal fundus image, OCT scan, or a clear close-up photo of the eye. Most common image formats are accepted (up to 20 MB).',
    details: ['JPEG / PNG / WEBP and more', 'Max file size: 20 MB', 'Drag-and-drop or click to upload'],
    color: 'bg-blue-500/10 text-blue-500',
    border: 'border-blue-500/20',
  },
  {
    step: '02',
    icon: Cpu,
    title: 'Image Preprocessing',
    description:
      'The image is resized to 384 × 384 pixels and normalised using ImageNet statistics (mean and std). This ensures the input exactly matches the distribution the model was trained on, maximising inference accuracy.',
    details: ['Resized to 384 × 384 px', 'ImageNet normalisation', 'Optimised for the CNN backbone'],
    color: 'bg-violet-500/10 text-violet-500',
    border: 'border-violet-500/20',
  },
  {
    step: '03',
    icon: Brain,
    title: 'Deep Learning Inference',
    description:
      'Your image passes through a state-of-the-art convolutional neural network trained on thousands of labelled retinal images. The model outputs a probability score for each of the 45 retinal disease classes simultaneously.',
    details: ['45-class multi-label CNN', 'Sigmoid output per class', 'GPU-accelerated inference'],
    color: 'bg-indigo-500/10 text-indigo-500',
    border: 'border-indigo-500/20',
  },
  {
    step: '04',
    icon: BarChart3,
    title: 'Threshold & Detection',
    description:
      'Each class probability is compared against a configurable threshold (default 0.5). Classes above the threshold are flagged as detected. You can lower the threshold to increase sensitivity or raise it for higher specificity.',
    details: ['Default threshold: 0.5', 'Adjustable per request', 'Controls sensitivity vs. specificity'],
    color: 'bg-cyan-500/10 text-cyan-500',
    border: 'border-cyan-500/20',
  },
  {
    step: '05',
    icon: ShieldCheck,
    title: 'Risk Assessment',
    description:
      'Detected diseases are cross-referenced against a curated list of high-risk conditions (e.g. Diabetic Retinopathy, CRVO, Vitreous Hemorrhage). The system assigns an overall risk level — HIGH, MODERATE, or LOW — based on the nature and number of findings.',
    details: ['HIGH / MODERATE / LOW risk levels', '12 high-priority disease flags', 'Aggregated across all findings'],
    color: 'bg-amber-500/10 text-amber-500',
    border: 'border-amber-500/20',
  },
  {
    step: '06',
    icon: FileText,
    title: 'Advisory & Report',
    description:
      'A clinical advisory note is generated for each detected condition, summarising recommended next steps. Results include all 45 confidence scores, detected conditions with full names, processing time, and a medical disclaimer.',
    details: ['Per-disease advisory notes', 'Full-name disease mapping', 'Processing time & disclaimer'],
    color: 'bg-emerald-500/10 text-emerald-500',
    border: 'border-emerald-500/20',
  },
];

const modelFacts = [
  { icon: Brain,        label: 'Architecture',  value: 'Multi-label CNN'   },
  { icon: Eye,          label: 'Disease Classes', value: '45 conditions'   },
  { icon: Zap,          label: 'Inference Speed', value: 'Sub-second'      },
  { icon: Activity,     label: 'Best AUC',        value: '0.8204'          },
  { icon: BarChart3,    label: 'Macro F1',         value: '0.1517'         },
  { icon: CheckCircle2, label: 'Input Resolution', value: '384 × 384 px'  },
];

const riskLevels = [
  {
    level: 'HIGH',
    variant: 'destructive',
    icon: AlertTriangle,
    desc: 'One or more critical conditions detected (e.g. DR, CRVO, VH, CRAO). Urgent ophthalmological review recommended.',
  },
  {
    level: 'MODERATE',
    variant: 'warning',
    icon: Activity,
    desc: 'Non-critical findings present. Follow-up with a clinician within a reasonable timeframe is advised.',
  },
  {
    level: 'LOW',
    variant: 'success',
    icon: CheckCircle2,
    desc: 'No significant findings above the detection threshold. Routine screening schedule can be maintained.',
  },
];

const stagger = {
  container: { animate: { transition: { staggerChildren: 0.1 } } },
  item: { initial: { opacity: 0, y: 20 }, animate: { opacity: 1, y: 0, transition: { duration: 0.45 } } },
};

export default function HowItWorks() {
  return (
    <PageTransition>
      <div className="mx-auto max-w-5xl px-4 sm:px-6 py-14 space-y-20">

        {/* Header */}
        <div className="text-center space-y-4 max-w-2xl mx-auto">
          <motion.span
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 rounded-full bg-primary/10 border border-primary/20 px-4 py-1.5 text-sm font-medium text-primary"
          >
            <Cpu className="h-3.5 w-3.5" />
            Pipeline Overview
          </motion.span>
          <motion.h1
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-4xl sm:text-5xl font-black tracking-tight"
          >
            How It <span className="text-primary">Works</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-muted-foreground text-lg leading-relaxed"
          >
            A step-by-step walkthrough of the AI pipeline — from image upload to
            clinical advisory output.
          </motion.p>
        </div>

        {/* Step-by-step pipeline */}
        <section className="space-y-6">
          {steps.map(({ step, icon: Icon, title, description, details, color, border }, i) => (
            <motion.div
              key={step}
              initial={{ opacity: 0, x: i % 2 === 0 ? -24 : 24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 0.45, delay: 0.05 }}
            >
              <Card className={`border ${border} overflow-hidden`}>
                <CardContent className="p-0">
                  <div className="flex flex-col sm:flex-row">
                    {/* Step number sidebar */}
                    <div className={`flex sm:flex-col items-center justify-center gap-3 px-6 py-5 sm:py-8 sm:w-24 ${color} shrink-0`}>
                      <span className="text-2xl font-black opacity-60">{step}</span>
                      <Icon className="h-7 w-7" />
                    </div>
                    {/* Content */}
                    <div className="flex-1 p-6 space-y-4">
                      <h3 className="text-xl font-bold text-foreground">{title}</h3>
                      <p className="text-sm text-muted-foreground leading-relaxed">{description}</p>
                      <div className="flex flex-wrap gap-2">
                        {details.map((d) => (
                          <span
                            key={d}
                            className="inline-flex items-center gap-1.5 text-xs bg-muted rounded-lg px-2.5 py-1 text-muted-foreground"
                          >
                            <CheckCircle2 className="h-3 w-3 text-primary shrink-0" />
                            {d}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </section>

        {/* Model facts */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-10"
          >
            <h2 className="text-2xl font-black text-foreground">Model at a Glance</h2>
            <p className="text-muted-foreground mt-1 text-sm">
              Key technical specifications of the underlying AI model.
            </p>
          </motion.div>

          <motion.div
            variants={stagger.container}
            initial="initial"
            whileInView="animate"
            viewport={{ once: true }}
            className="grid grid-cols-2 sm:grid-cols-3 gap-4"
          >
            {modelFacts.map(({ icon: Icon, label, value }) => (
              <motion.div key={label} variants={stagger.item}>
                <Card className="h-full hover:shadow-md transition-shadow">
                  <CardContent className="p-5 flex items-center gap-4">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10">
                      <Icon className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">{label}</p>
                      <p className="text-sm font-bold text-foreground">{value}</p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </section>

        {/* Risk level explanation */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-10"
          >
            <h2 className="text-2xl font-black text-foreground">Understanding Risk Levels</h2>
            <p className="text-muted-foreground mt-1 text-sm">
              How the system classifies the overall severity of findings.
            </p>
          </motion.div>

          <motion.div
            variants={stagger.container}
            initial="initial"
            whileInView="animate"
            viewport={{ once: true }}
            className="grid sm:grid-cols-3 gap-4"
          >
            {riskLevels.map(({ level, variant, icon: Icon, desc }) => (
              <motion.div key={level} variants={stagger.item}>
                <Card className="h-full hover:shadow-md transition-shadow">
                  <CardContent className="p-6 space-y-3">
                    <div className="flex items-center gap-2">
                      <Badge variant={variant}>
                        <Icon className="h-3 w-3 mr-1" />
                        {level}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed">{desc}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </section>

        {/* Disclaimer */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="flex items-start gap-3 rounded-xl bg-amber-500/5 border border-amber-500/20 px-5 py-4"
        >
          <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
          <p className="text-xs text-muted-foreground leading-relaxed">
            <span className="font-semibold text-foreground">Medical Disclaimer:</span>{' '}
            This system is designed for screening and educational purposes only. It is not a
            certified medical device and must not be used as a substitute for professional
            clinical diagnosis. Always consult a qualified ophthalmologist for interpretation
            and management of results.
          </p>
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <Card className="bg-gradient-to-r from-primary/10 via-primary/5 to-transparent border-primary/20">
            <CardContent className="p-8 sm:p-10 flex flex-col sm:flex-row items-center justify-between gap-6">
              <div>
                <h3 className="text-2xl font-black text-foreground">Ready to try it?</h3>
                <p className="text-muted-foreground mt-1">
                  Upload an eye image and see the full pipeline in action.
                </p>
              </div>
              <Link to="/predict" className="shrink-0">
                <Button size="lg" className="text-base">
                  <Eye className="h-5 w-5" />
                  Start Analysis
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>
        </motion.div>

      </div>
    </PageTransition>
  );
}
