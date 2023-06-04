import keras
import tensorflow as tf

class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_samples, fake_samples):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_samples - real_samples
        interpolated = real_samples + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_samples):
        if isinstance(real_samples, tuple):
            real_samples = real_samples[0]

        # Get the batch size
        batch_size = tf.shape(real_samples)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(1, self.latent_dim[0], self.latent_dim[1])
            )
            for j in range(8):
                with tf.GradientTape() as tape:
                    # Generate fake images from the latent vector
                    fake_samples = self.generator(random_latent_vectors, training=True)
                    next_random_latent_vectors = fake_samples
                    # merge random_latent_noise with fake_samples to get shape (batch_size, 2, 4000)
                    fake_samples = tf.concat([random_latent_vectors, fake_samples], axis=1)
                    # Get the logits for the fake images
                    fake_logits = self.discriminator(fake_samples, training=True)
                    # Get the logits for the real images
                    mini_batch_real_samples = real_samples[j]
                    mini_batch_real_samples = tf.reshape(mini_batch_real_samples, (1, 2, 4000))
                    real_logits = self.discriminator(mini_batch_real_samples, training=True)

                    # Calculate the discriminator loss using the fake and real image logits
                    d_cost = self.d_loss_fn(real_sample=real_logits, fake_sample=fake_logits)
                    # Calculate the gradient penalty
                    gp = self.gradient_penalty(1, mini_batch_real_samples, fake_samples)
                    # Add the gradient penalty to the original discriminator loss
                    d_loss = d_cost + gp * self.gp_weight
                    random_latent_vectors = next_random_latent_vectors
                # Get the gradients w.r.t the discriminator loss
                d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
                # Update the weights of the discriminator using the discriminator optimizer
                self.d_optimizer.apply_gradients(
                    zip(d_gradient, self.discriminator.trainable_variables)
                )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim[0], self.latent_dim[1]))
        for i in range(8):
            with tf.GradientTape() as tape:
                # Generate fake images using the generator
                generated_samples = self.generator(random_latent_vectors, training=True)
                next_random_latent_vectors = generated_samples
                # merge random_latent_noise with fake_samples to get shape (batch_size, 2, 4000)
                generated_samples = tf.concat([random_latent_vectors, generated_samples], axis=1)
                # Get the discriminator logits for fake images
                gen_samples_logits = self.discriminator(generated_samples, training=True)
                # Calculate the generator loss
                g_loss = self.g_loss_fn(gen_samples_logits)
                random_latent_vectors = next_random_latent_vectors
            # Get the gradients w.r.t the generator loss
            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(
                zip(gen_gradient, self.generator.trainable_variables)
            )
        return {"d_loss": d_loss, "g_loss": g_loss}
